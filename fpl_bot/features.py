from __future__ import annotations

import logging
import math
from enum import Enum
from functools import cached_property

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Small epsilon for numerical stability in validation checks.
_EPS = 1e-8


class FeatureType(Enum):
    GAUSSIAN = "gaussian"                 # symmetric continuous
    SKEWED_POSITIVE = "skewed_positive"   # heavy right tail
    SKEWED = "skewed"                     # heavy tail, but goes negative
    BIMODAL = "bimodal"                   # two peaks
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
    # external — loaded from CSV files
    VAASTAV = "vaastav"
    FCI = "fci"
    OPTA = "opta"
    FIXTURE = "fixture"
    # derived — computed internally by pipeline stages
    OPTAINGESTER = "opta_ingester"
    FPLINGESTER = "fpl_ingester"
    FIXINGESTER = "fixture_ingester"
    PRIOR = "prior"
    SEQUENCER = "sequencer"
    SCALER = "scaler"

EXTERNAL_DATA_SOURCES = frozenset({
    DataSource.VAASTAV,
    DataSource.FCI,
    DataSource.OPTA,
    DataSource.FIXTURE,
})


class FeatureSpec:
    """
    Specification for a single feature in the feature registry.

    Describes how a feature should be sourced, accumulated, scaled, and encoded.
    Used by the ingester, sequencer, and scaler to handle each feature correctly.
    """
    def __init__(
        self,
        name: str,
        feature_type: FeatureType,
        scaling_mode: ScalingMode,
        accumulation: AccumulationType,
        temporal: bool,
        source: dict[DataSource, str] | None = None,
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
            feature_type: Statistical type of the feature.
            scaling_mode: How the feature will be scaled before model input.
            accumulation: How the ingester accumulates this feature across gameweeks.
            temporal: Whether the feature varies over time.
            source: Mapping of DataSource to raw/output column name. External
                sources (VAASTAV, FCI, OPTA, FIXTURE) map to a raw CSV column;
                derived sources (INGESTER, SEQUENCER, PRIOR, SCALER) map to
                the output column produced by that pipeline stage. Every
                feature must have at least one entry.
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

        self.source = source if source is not None else {}

        # avoid mutable defaults
        self.scaling_params = scaling_params if scaling_params is not None else [None, None]

        self._validate()

    #================================================
    # Validation
    #================================================
 
    def _validate(self):
        """Validate feature spec fields, raising ValueError on misconfiguration."""
        # every feature must have at least one source mapping
        if not self.source:
            raise ValueError(
                f"{self.name}: source must be non-empty — "
                f"every feature needs at least one provider mapping."
            )

        # sequencer features must not be accumulated
        if DataSource.SEQUENCER in self.source:
            if self.accumulation != AccumulationType.NONE:
                raise ValueError(
                    f"{self.name}: SEQUENCER feature must have accumulation=NONE — "
                    f"sequencer features are not accumulated across gameweeks."
                )

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
            if abs(self.max_value) < _EPS:
                raise ValueError(f"max_value must be non-zero (abs < eps={_EPS}), received {self.max_value}.")

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

    @property
    def providers(self) -> set[DataSource]:
        """Set of all data sources (external and derived) for this feature."""
        return set(self.source.keys())

    @property
    def is_sequencer(self) -> bool:
        """True if this feature is stamped by the sequencer, not loaded from CSV."""
        return DataSource.SEQUENCER in self.source

    @property
    def is_derived(self) -> bool:
        """True if this feature is computed internally rather than loaded from CSV."""
        return not self.providers.issubset(EXTERNAL_DATA_SOURCES)


class Features:
    """
    Ordered, immutable collection of FeatureSpec objects.

    Provides prebuilt tensor masks and column name lists used throughout
    the ingestion, scaling, and encoding pipeline.
    """

    def __init__(self, specs: list[FeatureSpec]):
        """
        Initialise Features from a list of FeatureSpec objects.

        Specs are auto-sorted so all non-categorical features come first,
        followed by all categorical features. Relative order within each
        group is preserved.

        Args:
            specs: List of feature specifications. Will be reordered so
                categoricals come last before freezing.
        """
        # auto-sort: numeric specs first, categorical specs last
        numeric = [s for s in specs if s.feature_type != FeatureType.CATEGORICAL]
        categorical = [s for s in specs if s.feature_type == FeatureType.CATEGORICAL]
        self.specs = tuple(numeric + categorical)

        self._validate()

        # prebuild lookup
        self._spec_by_name: dict[str, FeatureSpec] = {s.name: s for s in self.specs}

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
            if not s.is_sequencer
        ]

    @cached_property
    def sequencer_columns(self) -> list[str]:
        """Feature names stamped by the sequencer at sequence-build time; not loaded from CSV."""
        return [
            s.name for s in self.specs
            if s.is_sequencer
        ]

    @cached_property
    def numeric_columns(self) -> list[str]:
        """Feature names for numeric (non-categorical) features; defines the scaled tensor columns."""
        return [s.name for s in self.specs if s.feature_type != FeatureType.CATEGORICAL]

    @cached_property
    def categorical_columns(self) -> list[str]:
        """Feature names for categorical features; fed into nn.Embedding as integer indices."""
        return [s.name for s in self.specs if s.feature_type == FeatureType.CATEGORICAL]

    @cached_property
    def numeric_indices(self) -> list[int]:
        """Indices into specs for non-CATEGORICAL features."""
        return [i for i, s in enumerate(self.specs) if s.feature_type != FeatureType.CATEGORICAL]

    @cached_property
    def categorical_indices(self) -> list[int]:
        """Indices into specs for CATEGORICAL features."""
        return [i for i, s in enumerate(self.specs) if s.feature_type == FeatureType.CATEGORICAL]

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
    # Provide Filtered Instance
    # returns instance of features, filtered by condition
    #================================================

    @property
    def filtered_categoric(self) -> "Features":
        """
        Provides new instance of only categoric features.
        
        Used to select features that require embedding.

        Returns:
            Instance of Features, for only categorics
        """
        return Features(
            [self._spec_by_name[name] for name in self.categorical_columns]
        )

    @property    
    def filtered_numeric(self) -> "Features":
        """
        Provides new instance of only numeric features.
        
        Used to select features that require scaling.

        Returns:
            Instance of Features, for only numerics
        """
        return Features(
            [self._spec_by_name[name] for name in self.numeric_columns]
        )
        
    #================================================
    # Provider-Filtered Column Lists
    # Column lists narrowed to features supplied by a specific DataSource.
    #================================================

    def _columns_for(self, base_list: list[str], providers: DataSource | list[DataSource]) -> list[str]:
        """
        Helper: Filter any column list to only those supplied by given providers.

        Args:
            base_list: List of output column names to filter.
            providers: Single DataSource or list of DataSources to filter by.

        Returns:
            Subset of base_list where any provider has a non-None source column.
        """
        providers = self._normalise_providers(providers)
        return [
            name for name in base_list
            if self._has_provider(self._spec_by_name[name], providers)
        ]

    def output_columns_for(self, providers: list[DataSource]) -> list[str]:
        """
        Pre-Sequencer column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Source column names available for that provider.
        """
        return self._columns_for(self.output_columns, providers)

    def categorical_columns_for(self, providers: list[DataSource]) -> list[str]:
        """
        Categorical output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Categorical column names available for that provider.
        """
        return self._columns_for(self.categorical_columns, providers)

    def cumulative_columns_for(self, providers: list[DataSource]) -> list[str]:
        """
        Cumulative output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Cumulative column names available for that provider.
        """
        return self._columns_for(self.cumulative_columns, providers)

    def per_90_columns_for(self, providers: list[DataSource]) -> list[str]:
        """
        Per-90-rate output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Per-90 column names available for that provider.
        """
        return self._columns_for(self.per_90_columns, providers)

    def snapshot_columns_for(self, providers: list[DataSource]) -> list[str]:
        """
        Snapshot output column names supplied by a given provider.

        Args:
            provider: DataSource to filter by.

        Returns:
            Snapshot column names available for that provider.
        """
        return self._columns_for(self.snapshot_columns, providers)

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

    def cumulative_map_for(self, providers: DataSource | list[DataSource]) -> dict[str, str]:
        """Dict mapping output column name to cumulative column name for all accumulated features from given providers."""
        cols = self.output_columns_for(providers)
        return {
            col: self._spec_by_name[col].cum_col
            for col in cols
            if self._spec_by_name[col].cum_col is not None
        }

    def inv_cumulative_map_for(self, providers: DataSource | list[DataSource]) -> dict[str, str]:
        """Dict mapping cumulative column name to output column name (inverse of cumulative_map) from given providers."""
        return {
            cum: out
            for out, cum in self.cumulative_map_for(providers).items()
        }

    def source_map(self, provider: DataSource) -> dict[str, str]:
        """
        Maps source columns to output columns.

        Args:
            provider: Dictates which provider to source columns from.

        Returns:
            Dict: keys are source names, values are output names.
        """
        return {
            s.source[provider]: s.name
            for s in self.specs
            if provider in s.source
        }

    def get_source_names(self, base_list: list[str], providers: DataSource | list[DataSource]) -> list[str]:
        """
        Given a list of features names, and provider(s). Maps to source column names.

        Args:
            providers: Single DataSource or list of DataSources to filter by.

        Returns:
            List of features source name.
        """
        providers = self._normalise_providers(providers)
        result = []
        for name in base_list:
            spec = self._spec_by_name[name]
            
            for provider in providers:
                if provider in spec.source:
                    result.append(spec.source[provider])
    
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

    def index_from(self, index: int) -> str:
        """
        Return name of feature at index.

        Args:
            index: requested index

        Returns: 
            string name of feature at index

        Raises:
            KeyError: if no feature of index
        """

        return self.specs[index].name
        
    @property
    def specs_by_mode(self) -> dict[ScalingMode, list[FeatureSpec]]:
        """Group specs by their scaling mode."""
        return {mode: [s for s in self.specs if s.scaling_mode == mode] for mode in ScalingMode}

    @property
    def specs_by_provider(self) -> dict[DataSource, list[FeatureSpec]]:
        """Group specs by provider; a multi-provider feature appears under each."""
        result: dict[DataSource, list[FeatureSpec]] = {}
        for s in self.specs:
            for provider in s.source:
                result.setdefault(provider, []).append(s)
        return result

    #================================================
    # DataFrame Validation
    # Runtime checks on DataFrames against the feature spec.
    #================================================

    def validate_dataframe_from_source(
        self, 
        df: pd.DataFrame, 
        feature_list: list[str], 
        providers: DataSource | list[DataSource] , 
        context: str = ""
    ) -> None:
        """
        Takes list of Feature names, and checks if source names are in df (i.e apply before renaming).

        Raises ValueError if columns are missing; logs a warning for extra columns.

        Args:
            df: DataFrame to validate.
            feature_list: Input list of feature names.
            context: Optional label for error messages, e.g. "GW1".
        """
        # convert to source names 
        expected_cols = self.get_source_names(feature_list, providers)
        expected = set(expected_cols)
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"{context} missing columns: {sorted(missing)}")

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
                "feature_type": s.feature_type.value,
                "scaling_mode": s.scaling_mode.value,
                "accumulation": s.accumulation.value,
                "scaling_params": s.scaling_params,
                "temporal": s.temporal,
                "source": {k.value: v for k, v in s.source.items()} if s.source else {},
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
            # backwards compat: accept old "source_columns" key as well as "source"
            raw_source = d.get("source", d.get("source_columns", {}))

            # backwards compat: old "seq" value maps to new SEQUENCER
            source = {}
            for k, v in raw_source.items():
                ds_value = "sequencer" if k == "seq" else k
                source[DataSource(ds_value)] = v

            # backwards compat: merge any old "derived" entries into source
            for k, v in d.get("derived", {}).items():
                source.setdefault(DataSource(k), v)

            # backwards compat: old JSON may have feature_provider instead
            if not source and "feature_provider" in d:
                provider_str = d["feature_provider"]
                ds_value = "sequencer" if provider_str == "seq" else provider_str
                source[DataSource(ds_value)] = d["name"]

            specs.append(
                FeatureSpec(
                    name=d["name"],
                    feature_type=FeatureType(d["feature_type"]),
                    scaling_mode=ScalingMode(d["scaling_mode"]),
                    accumulation=AccumulationType(d.get("accumulation", "none")),
                    scaling_params=d["scaling_params"],
                    temporal=d["temporal"],
                    source=source,
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
    #================================================

    def temporal_mask(self) -> torch.Tensor:
        """
        Build boolean mask selecting temporal features.
 
        Returns:
            Boolean mask true for each temporal feature.
        """
        return torch.tensor([s.temporal for s in self.specs])

    def build_scaling_masks(self) -> dict[ScalingMode, torch.Tensor]:
        """
        Builds boolean masks for each scaling mode.
        
        Returns:
            Scaling masks for each scaling mode; dict[ScalingMode, Tensor]
        """
        masks = {}
        for mode in ScalingMode:
            masks[mode] = torch.tensor(
                [s.scaling_mode == mode for s in self.specs],
                dtype=torch.bool
            )
        return masks

    #================================================
    # Private Helpers
    #================================================

    @staticmethod
    def _normalise_providers(providers: DataSource | list[DataSource]) -> list[DataSource]:
        """Wrap a single DataSource in a list for uniform iteration."""
        if isinstance(providers, DataSource):
            return [providers]
        return providers

    @staticmethod
    def _has_provider(spec: FeatureSpec, providers: list[DataSource]) -> bool:
        """True if spec has a non-None source entry for any of the given providers."""
        return any(
            p in spec.source and spec.source[p] is not None
            for p in providers
        )

    #================================================
    # Dunder Overrides
    #================================================

    def __len__(self) -> int:
        """Return the number of features in this collection."""
        return len(self.specs)
