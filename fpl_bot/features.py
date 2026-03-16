from __future__ import annotations

import logging
import math
from enum import Enum
from functools import cached_property

import pandas as pd
import torch

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    GAUSSIAN = "gaussian"                 # symmetric continuous
    SKEWED_POSITIVE = "skewed_positive"   # heavy right tail
    COUNT = "count"                       # sparse non-negative integers
    BOUNDED = "bounded"                   # known min/max
    BINARY = "binary"                     # 0/1
    ORDINAL = "ordinal"                   # ordered categories
    CATEGORICAL = "categorical"           # nominal categories
    PERIODIC = "periodic"                 # periodic numeric
    IDENTITY = "identity"                 # leave untouched

class ScalingMode(Enum):
    LINEAR = 'linear'
    LOG = 'log'
    ROBUST = 'robust'
    BOUNDED = 'bounded'
    LOG_ROBUST = 'log_robust'
    IDENTITY = 'identity'

class AccumulationType(Enum):
    NONE = "none"           # snapshot features, not accumulated
    PER_90 = "per_90"       # accumulative and then divided by cum_minutes / 90
    RAW_CUMULATIVE = "raw"  # accumulated but NOT divided by minutes

class DataSource(Enum):
    VAASTAV = "vaastav"
    FCI = "fci"
    OPTA = "opta"
    FIXTURE = "fixture"
    SEQUENCER = "seq"

EXTERNAL_DATA_SOURCES = frozenset({DataSource.VAASTAV, DataSource.FCI, DataSource.OPTA, DataSource.FIXTURE})


class FeatureSpec:
    """
    Specification for a single feature in the feature registry.

    Describes how a feature should be sourced, accumulated, scaled, and encoded.
    Used by the ingester, sequencer, and scaler to handle each feature correctly.
    """
    def __init__(
        self,
        name: str,
        feature_provider: DataSource,
        feature_type: FeatureType,
        scaling_mode: ScalingMode,
        accumulation: AccumulationType,
        temporal: bool,
        source_columns: dict[DataSource, str] | None = None,
        presence_check: bool = False,
        categories: list | None = None,
        embedding_dim: int | None = None,
        period: int | None = None,
        max_value: float | None = None,
        min_value: float = 0,
        scaling_params: list | None = None,
    ):
        """
        Initialise a FeatureSpec.

        Args:
            name: Feature name, used as the global output column identifier.
            feature_provider: Data source this feature originates from.
            feature_type: Statistical type of the feature.
            scaling_mode: How the feature will be scaled before model input.
            accumulation: How the ingester accumulates this feature across gameweeks.
            temporal: Whether the feature varies over time.
            source_columns: Mapping of provider to raw column name,
                e.g. {DataSource.VAASTAV: "goals_scored"}.
            presence_check: If True, a feature value equal to min_value indicates
                missing data rather than a genuine zero.
            categories: Category list for categorical features.
            embedding_dim: Embedding dimension for categorical features.
            period: Period for periodic (cyclic) features.
            max_value: Maximum value for bounded scaling.
            min_value: Minimum value for bounded scaling or presence check threshold.
            scaling_params: Fitted scaling parameters [p1, p2]; populated after scaling.
        """
        self.name = name
        self.feature_provider = feature_provider
        self.feature_type = feature_type
        self.scaling_mode = scaling_mode
        self.temporal = temporal
        self.accumulation = accumulation

        self.presence_check = presence_check
        self.categories = categories
        self.embedding_dim = embedding_dim
        self.max_value = max_value
        self.min_value = min_value
        self.period = period

        # avoid mutable defaults
        # {DataSource: raw_column_name}
        self.source_columns = source_columns if source_columns is not None else {}
        # fitted scaling parameters [p1, p2]
        self.scaling_params = scaling_params if scaling_params is not None else [None, None]

        self._validate()

    #================================================
    # Validation
    #================================================
 
    def _validate(self):
        """Validate feature spec fields, raising ValueError on misconfiguration."""
        # periodic features require given period
        if self.feature_type == FeatureType.PERIODIC:
            if self.period is None:
                raise ValueError(f"{self.name} requires period.")

        if self.feature_type == FeatureType.BOUNDED:
            # bounded features require max, else if max not in data cannot calculate
            if self.max_value is None:
                raise ValueError(f"{self.name} requires a max_value.")
            # finite max_value
            if not math.isfinite(self.max_value):
                raise ValueError(f"max_value must be finite, received {self.max_value}.")
            # max_value must be non-zero
            if abs(self.max_value) < self.eps:
                raise ValueError(f"max_value must be non-zero (abs < eps={self.eps}), received {self.max_value}.")

        # categorical feature requires categories
        if self.feature_type == FeatureType.CATEGORICAL:
            if self.categories is None:
                raise ValueError(f"{self.name} requires categories.")

        # a feature that implies data was present (i.e 0 means bad performance not missing) requires threshold
        if self.presence_check:
            if self.min_value is None:
                raise ValueError(f"{self.name} requires min_value to be presence check")

    #================================================
    # Accumulation Properties
    # Derived properties for querying accumulation type.
    #================================================

    @property
    def cum_col(self) -> str | None:
        """
        Derives the cumulative column name from the feature name.

        PER_90: "goals_per_90" → "cum_goals"; RAW_CUMULATIVE: "minutes" → "cum_minutes"; NONE: None.
        """
        if self.accumulation == AccumulationType.NONE:
            return None
        if self.accumulation == AccumulationType.PER_90:
            return "cum_" + self.name.replace("_per_90", "")
        # RAW_CUMULATIVE
        return "cum_" + self.name

    @property
    def is_snapshot(self) -> bool:
        """True if this feature is a snapshot (not accumulated across gameweeks)."""
        return self.accumulation == AccumulationType.NONE

    @property
    def is_per_90(self) -> bool:
        """True if this feature is accumulated and then divided by minutes per 90."""
        return self.accumulation == AccumulationType.PER_90

    @property
    def is_cumulative(self) -> bool:
        """True if this feature is accumulated across gameweeks (PER_90 or RAW_CUMULATIVE)."""
        return self.accumulation != AccumulationType.NONE


class Features:
    """
    Ordered, immutable collection of FeatureSpec objects.

    Provides prebuilt tensor masks and column name lists used throughout
    the ingestion, scaling, and encoding pipeline.
    """

    def __init__(self, specs: list[FeatureSpec], eps=1e-8):
        """
        Initialise Features from a list of FeatureSpec objects.

        Args:
            specs: Ordered list of feature specifications. Order determines tensor column order.
            eps: Small epsilon used for numerical stability in validation.
        """
        # tuple makes immutable after initialisation
        self.specs = tuple(specs)

        self.eps = eps
        self._validate()

        # prebuild lookup
        self._spec_by_name: dict[str, FeatureSpec] = {s.name: s for s in self.specs}
        # tensor masks
        self.scaling_masks = self._build_mode_masks()       # builds tensor, a mask for each scaling mode, shape = [n_features]
        self.type_masks = self._build_type_mask()           # builds tensor, a mask for each feature type, shape = [n_features]
        self.temporal_mask = self._temporal_mask()          # builds tensor, a temporal mask, True is temporal, shape = [n_features]

    #================================================
    # Validation
    # Validates feature specs on initialisation.
    #================================================

    def _validate(self):
        """Validate all feature specs, raising ValueError on misconfiguration."""
        # validate input
        # ensure no duplicates
        names = self.output_columns
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names detected.")

        cum_cols = []
        for s in self.specs:
       # --- Pipeline Wiring Guards ---

            # every data-sourced feature must have source_columns wired up.
            if s.feature_provider in EXTERNAL_DATA_SOURCES and not s.source_columns:
                raise ValueError(
                    f"{s.name}: non-SEQUENCER feature (provider={s.feature_provider.value}) "
                    f"must have source_columns mapping — otherwise the ingester cannot load it."
                )

            # sequencer features must NOT have source_columns or accumulation.
            if s.feature_provider == DataSource.SEQUENCER:
                if s.source_columns:
                    raise ValueError(
                        f"{s.name}: SEQUENCER feature must not have source_columns — "
                        f"it is stamped at sequence-build time, not loaded from CSV."
                    )
                if s.accumulation != AccumulationType.NONE:
                    raise ValueError(
                        f"{s.name}: SEQUENCER feature must have accumulation=NONE — "
                        f"sequencer features are not accumulated across gameweeks."
                    )

            # cumulative column uniqueness
            if s.cum_col is not None:
               cum_cols.append(s.cum_col)

        if len(cum_cols) != len(set(cum_cols)):
            raise ValueError("Duplicate cumulative column names derived from specs.")

    #================================================
    # Column Name Lists
    # Properties returning lists of column names for tensor
    # indexing, ingestion, and accumulation logic.
    # Cached as specs is a frozen tuple after initialisation.
    #================================================

    @cached_property
    def output_columns(self) -> list[str]:
        """Ordered list of all feature names; defines tensor column order for scaling and model input."""
        return [s.name for s in self.specs]

    @cached_property
    def pre_sequencer_columns(self) -> list[str]:
        """Feature names present before sequencer stamping, excluding SEQUENCER-sourced features."""
        return [
            s.name for s in self.specs
            if s.feature_provider != DataSource.SEQUENCER
        ]

    @cached_property
    def sequencer_columns(self) -> list[str]:
        """Feature names stamped by the sequencer at sequence-build time; not loaded from CSV."""
        return [
            s.name for s in self.specs
            if s.feature_provider == DataSource.SEQUENCER
        ]

    @cached_property
    def fixture_columns(self) -> list[str]:
        """Feature names that come from fixture information"""
        return [s.name for s in self.specs if s.feature_provider == DataSource.FIXTURE]

    @cached_property
    def temporal_columns(self) -> list[str]:
        """Feature names that feed into x_temporal; order matches self.specs for stable tensor indexing."""
        return [s.name for s in self.specs if s.temporal]

    @cached_property
    def categorical_columns(self) -> list[str]:
        """Feature names that feed into x_categorical as embedded integer indices."""
        return [s.name for s in self.specs if not s.temporal]

    @cached_property
    def snapshot_columns(self) -> list[str]:
        """Feature names with AccumulationType.NONE; not accumulated across gameweeks."""
        return [s.name for s in self.specs if s.is_snapshot]

    @cached_property
    def per_90_columns(self) -> list[str]:
        """Feature names with AccumulationType.PER_90; expressed as per-90-minute rates."""
        return [s.name for s in self.specs if s.is_per_90]

    @cached_property
    def cumulative_columns(self) -> list[str]:
        """Feature names accumulated across gameweeks (PER_90 or RAW_CUMULATIVE)."""
        return [s.name for s in self.specs if s.is_cumulative]

    @cached_property
    def raw_cumulative_columns(self) -> list[str]:
        """Feature names accumulated but not per-90 divided; typically 'minutes' and 'featured'."""
        return [s.name for s in self.specs if s.accumulation == AccumulationType.RAW_CUMULATIVE]

    #================================================
    # Provider-Filtered Column Lists
    # Column lists narrowed to features supplied by a specific DataSource.
    #================================================

    def _columns_for(self, base_list: list[str], provider: DataSource) -> list[str]:
        """
        Helper: Filter any column list to only those supplied by a given provider.

        Args:
            base_list: List of output column names to filter.
            provider: DataSource to filter by.

        Returns:
            Subset of base_list where the provider has a non-None source column.
        """
        return [
            name for name in base_list
            if provider in self._spec_by_name[name].source_columns
            and self._spec_by_name[name].source_columns[provider] is not None
        ]

    def source_columns_for(self, provider: DataSource) -> list[str]:
        """
        Source columns names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Source column names available for that provider.
        """
        return [
            s.source_columns[provider] for s in self.specs
            if provider in s.source_columns
            and s.source_columns[provider] is not None
        ]

    def categorical_columns_for(self, provider: DataSource) -> list[str]:
        """
        Categorical output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Categorical column names available for that provider.
        """
        return self._columns_for(self.categorical_columns, provider)

    def cumulative_columns_for(self, provider: DataSource) -> list[str]:
        """
        Cumulative output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Cumulative column names available for that provider.
        """
        return self._columns_for(self.cumulative_columns, provider)

    def per_90_columns_for(self, provider: DataSource) -> list[str]:
        """
        Per-90-rate output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Per-90 column names available for that provider.
        """
        return self._columns_for(self.per_90_columns, provider)

    def snapshot_columns_for(self, provider: DataSource) -> list[str]:
        """
        Snapshot output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Snapshot column names available for that provider.
        """
        return self._columns_for(self.snapshot_columns, provider)

    #================================================
    # Column Mapping Dicts
    # Dictionaries mapping between output names, cumulative names,
    # and raw source column names.
    #================================================

    @property
    def cumulative_map(self) -> dict[str, str]:
        """Dict mapping output column name to cumulative column name for all accumulated features."""
        return {
            s.name: s.cum_col
            for s in self.specs
            if s.cum_col is not None
        }

    @property
    def inv_cumulative_map(self) -> dict[str, str]:
        """Dict mapping cumulative column name to output column name (inverse of cumulative_map)."""
        return {
            cum: out
            for out, cum in self.cumulative_map.items()
        }

    def cumulative_map_for(self, provider: DataSource) -> dict[str, str]:
        """
        Maps output column name to cumulative column name for a given provider.

        Args:
            provider: DataSource to filter by.
        """
        return {
            s: self._spec_by_name[s].cum_col
            for s in self.cumulative_map
            if provider in self._spec_by_name[s].source_columns
            and self._spec_by_name[s].source_columns[provider] is not None
        }

    def inv_cumulative_map_for(self, provider: DataSource) -> dict[str, str]:
        """
        Maps cumulative column name to output column name for a given provider.

        Args:
            provider: DataSource to filter by.
        """
        return {
            self._spec_by_name[s].cum_col: s
            for s in self.cumulative_map
            if provider in self._spec_by_name[s].source_columns
            and self._spec_by_name[s].source_columns[provider] is not None
        }

    def source_map(self, provider: DataSource) -> dict[str, str]:
        """
        Maps raw source column name to output column name for one data provider.

        Args:
            provider: DataSource key into FeatureSpec.source_columns,
                e.g. DataSource.VAASTAV, DataSource.FCI, DataSource.OPTA.

        Returns:
            Dict of {source_col: output_col} for all specs with this provider.
        """
        result = {}
        for s in self.specs:
            if s.source_columns and provider in s.source_columns:
                source_col = s.source_columns[provider]
                if source_col is not None:
                    result[source_col] = s.name
        return result

    #================================================
    # Spec Lookups
    # Access individual specs by name, index, or grouped
    # by mode or provider.
    #================================================

    def __getitem__(self, name: str) -> FeatureSpec:
        """
        Look up a FeatureSpec by name.

        Args:
            name: Feature name to look up.

        Returns:
            The matching FeatureSpec.

        Raises:
            KeyError: If no feature with that name exists.
        """
        try:
            return self._spec_by_name[name]
        except KeyError:
            raise KeyError(f"No feature named '{name}'. Available: {list(self._spec_by_name.keys())}")

    def __contains__(self, name: str) -> bool:
        """Return True if a feature with the given name exists in this collection."""
        return name in self._spec_by_name

    def index_of(self, name: str) -> int:
        """
        Return the tensor column index of a named feature.

        Args:
            name: Feature name to look up.

        Returns:
            Integer index of the feature in the output tensor.

        Raises:
            KeyError: If no feature with that name exists.
        """
        for i, s in enumerate(self.specs):
            if s.name == name:
                return i
        raise KeyError(f"No feature named '{name}'")

    @property
    def specs_by_mode(self) -> dict[ScalingMode, list[FeatureSpec]]:
        """Group specs by their scaling mode."""
        return {mode: [s for s in self.specs if s.scaling_mode == mode] for mode in ScalingMode}

    @property
    def specs_by_provider(self) -> dict[DataSource, list[FeatureSpec]]:
        """Group specs by their feature_provider field."""
        result: dict[DataSource, list[FeatureSpec]] = {}
        for s in self.specs:
            result.setdefault(s.feature_provider, []).append(s)
        return result

    #================================================
    # DataFrame Validation
    # Runtime checks on DataFrames against the feature spec.
    #================================================

    def validate_dataframe_from(self, df: pd.DataFrame, provider: DataSource, context: str = "") -> None:
        """
        Check that a DataFrame from a provider contains the expected columns.

        Raises ValueError if columns are missing; logs a warning for extra columns.

        Args:
            df: DataFrame to validate.
            provider: DataSource whose source columns define the expected set.
            context: Optional label for error messages, e.g. "GW1".
        """
        expected = set(self.source_columns_for(provider))
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"{context} missing columns: {sorted(missing)}")
        if extra:
            logger.warning(f"{context} has extra columns not in Features: {sorted(extra)}")

    #================================================
    # Serialisation
    # Convert to and from a JSON-serialisable dict format.
    #================================================

    def to_dict(self) -> list[dict]:
        """
        Serialise all feature specs to a list of dicts.

        Returns:
            List of dicts, one per spec, suitable for JSON serialisation.
        """
        return [
            {
                "name": s.name,
                "feature_provider": s.feature_provider.value,
                "feature_type": s.feature_type.value,
                "scaling_mode": s.scaling_mode.value,
                "accumulation": s.accumulation.value,
                "scaling_params": s.scaling_params,
                "temporal": s.temporal,
                "source_columns": {k.value: v for k, v in s.source_columns.items()} if s.source_columns else {},
                "presence_check": s.presence_check,
                "categories": s.categories,
                "embedding_dim": s.embedding_dim,
                "period": s.period,
                "max_value": s.max_value,
                "min_value": s.min_value,
            }
            for s in self.specs
        ]

    @classmethod
    def from_dict(cls, data: list[dict]) -> "Features":
        """
        Deserialise a Features object from a list of dicts.

        Args:
            data: List of spec dicts as produced by to_dict().

        Returns:
            Reconstructed Features instance.
        """
        specs = []
        for d in data:
            specs.append(
                FeatureSpec(
                    name=d["name"],
                    feature_provider=DataSource(d["feature_provider"]),
                    feature_type=FeatureType(d["feature_type"]),
                    scaling_mode=ScalingMode(d["scaling_mode"]),
                    accumulation=AccumulationType(d.get("accumulation", "none")),
                    scaling_params=d["scaling_params"],
                    temporal=d["temporal"],
                    source_columns={DataSource(k): v for k, v in d.get("source_columns", {}).items()},
                    presence_check=d["presence_check"],
                    categories=d["categories"],
                    embedding_dim=d["embedding_dim"],
                    period=d["period"],
                    max_value=d["max_value"],
                    min_value=d["min_value"],
                )
            )
        return cls(specs)

    #================================================
    # Tensor Mask Builders
    # Private methods called once at initialisation to create boolean
    # tensors used by the scaling and encoding layers.
    #================================================

    def _temporal_mask(self) -> torch.Tensor:
        """Build boolean mask selecting temporal features."""
        return torch.tensor([s.temporal for s in self.specs])

    def _build_mode_masks(self) -> dict[ScalingMode, torch.Tensor]:
        """Build boolean masks for each scaling mode."""
        masks = {}
        for mode in ScalingMode:
            masks[mode] = torch.tensor(
                [s.scaling_mode == mode for s in self.specs],
                dtype=torch.bool
            )
        return masks

    def _build_type_mask(self) -> dict[FeatureType, torch.Tensor]:
        """Build boolean masks for each feature type."""
        masks = {}
        for ftype in FeatureType:
            masks[ftype] = torch.tensor(
                [s.feature_type == ftype for s in self.specs],
                dtype=torch.bool
            )
        return masks

    #================================================
    # Dunder Overrides
    #================================================

    def __len__(self) -> int:
        """Return the number of features in this collection."""
        return len(self.specs)
