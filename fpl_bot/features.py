import torch
import logging
from enum import Enum  
import math
import pandas as pd
from functools import cached_property
logging.basicConfig(level=logging.INFO)
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
    SEQUENCER = "seq"

class FeatureSpec:
 
    def __init__(
        self,
        name: str,
        feature_provider: DataSource,
        feature_type: FeatureType,
        scaling_mode: ScalingMode,
        accumulation: AccumulationType,
        temporal: bool,
        source_columns: dict[DataSource, str] = {},
        presence_check: bool = False,
        categories: list | None = None,
        embedding_dim: int | None = None,
        period: int | None = None,
        max_value: float | None = None,
        min_value: float = 0,        
        scaling_params: list | None = None,
    ):
        self.name = name                                # feature name, output_name (this is global convention for names of features)
        self.feature_provider = feature_provider        # which data set feature orignates from "opta" or "fpl"
        self.feature_type = feature_type                # data type of feature
        self.scaling_mode = scaling_mode                # how feature will be scaled
        self.temporal = temporal                        # does feature vary with time
        self.accumulation = accumulation                # how ingester accumulates this across GWs
        self.source_columns = source_columns            # {provider_name: raw_column_name} e.g. {"vaastav": "goals_scored", "fci": "goals_scored"}
        self.presence_check = presence_check            # if true, any sample with feature value = min value is missing from data NOT 0
        self.categories = categories                    # if categorical type, what catergories
        self.embedding_dim = embedding_dim              # if embedded, how many dims
        self.max_value = max_value                      # max value of feature, for bounded
        self.min_value = min_value                      # min value of feature, for bounded and presence check
        self.period = period                            # period of cyclic feature
 
        # parameters fit from scaling 
        self.scaling_params = scaling_params if scaling_params is not None else [None, None]                 
 
    # ══════════════════════════════════════════════════════════════════
    #  Accumulation Properties
    # ══════════════════════════════════════════════════════════════════
 
    @property
    def cum_col(self) -> str | None:
        """
        Givens cumulative column name for feature
        Convention:
            PER_90:         "goals_per_90" → "cum_goals"
            RAW_CUMULATIVE: "minutes"      → "cum_minutes"
            NONE:           returns None
        """
        if self.accumulation == AccumulationType.NONE:
            return None
        if self.accumulation == AccumulationType.PER_90:
            return "cum_" + self.name.replace("_per_90", "")
        # RAW_CUMULATIVE
        return "cum_" + self.name
    
    @property 
    def is_snapshot(self) -> bool:
        return self.accumulation == AccumulationType.NONE
 
    @property
    def is_per_90(self) -> bool:
        return self.accumulation == AccumulationType.PER_90
 
    @property
    def is_cumulative(self) -> bool:
        return self.accumulation != AccumulationType.NONE    
 
 
class Features:
 
    def __init__(self, specs: list[FeatureSpec], eps=1e-8):
        # tuple makes immutable after initialisation
        self.specs = tuple(specs)
 
        self.eps = eps
        self._validate()
 
        # prebuild lookup 
        self._spec_by_name: dict[str, FeatureSpec] = {s.name: s for s in self.specs}
        # tensor masks
        self.scaling_masks = self._build_mode_masks()       # builds tensor, a mask for each scaling mode, shape = [n_features]
        self.type_masks = self._build_type_mask()           # builds tensor, a mask for each feature type, shape = [ n_features]
        self.temporal_mask = self._temporal_mask()          # builds tensor, a temporal mask, True is tempooral, shape = [n_features]
 
    # ══════════════════════════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════════════════════════
 
    def _validate(self):
        # validate input
        # ensure no duplicates
        names = self.output_columns
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names detected.")
 
        for s in self.specs:
            # periodic features require given period
            if s.feature_type == FeatureType.PERIODIC:
                if s.period is None:
                    raise ValueError(f"{s.name} requires period.")
 
            if s.feature_type == FeatureType.BOUNDED:
                # bound features require max, else if max not in data cannot calculate
                if s.max_value is None:
                    raise ValueError(f"{s.name} requires a max_value.")
                # finite max_value
                if not math.isfinite(s.max_value):
                    raise ValueError(f"max_value must be finite, received {s.max_value}.")
                # max_value must be non-zero
                if abs(s.max_value) < self.eps:
                    raise ValueError(f"max_value must be non-zero (abs < eps={self.eps}), received {s.max_value}.")
            # categorical feature requires catergories
            if s.feature_type == FeatureType.CATEGORICAL:
                if s.categories is None:
                    raise ValueError(f"{s.name} requires catergories")
            # a feature that implies data was present (i.e 0 means bad performance not missing) requires threshold
            if s.presence_check == True:
                if s.min_value is None:
                    raise ValueError(f"{s.name} requires min_value to be presence check")
 
       # --- Pipeline wiring guards ---
 
            # every data-sourced feature must have source_columns wired up.
            if s.feature_provider in set({DataSource.VAASTAV, DataSource.FCI, DataSource.OPTA}) and not s.source_columns:
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
            cum_cols = []
            if s.cum_col is not None:
                cum_cols = cum_cols.append(s.cum_col) 

        if len(cum_cols) != len(set(cum_cols)):
            raise ValueError("Duplicate cumulative column names derived from specs.")
    # ══════════════════════════════════════════════════════════════════
    #  Column Name Lists
    #  Properties that return lists of column names, used for tensor
    #  indexing, ingestion, and accumulation logic.
    #  cahched as built from tuple so immutable after init
    # ══════════════════════════════════════════════════════════════════
 
    @cached_property
    def output_columns(self) -> list[str]:
        """
        This method defines global truth for features leaving sequencer, i.e for scaling and feeding to model.
        Immutable as specs is frozen as tuple.
        """
        return [s.name for s in self.specs]

    @cached_property
    def pre_sequencer_columns(self) -> list[str]:
        """
        This method controls feature order before sequencer.
        Immutable as specs is frozen as tuple.
        """
        return [
            s.name for s in self.specs
            if s.feature_provider != DataSource.SEQUENCER
        ]

    @cached_property
    def sequencer_columns(self) -> list[str]:
        """
        These are columns the sequencer stamps on every row it builds.
        """
        return [
            s.name for s in self.specs
            if s.feature_provider == DataSource.SEQUENCER
        ]

    @cached_property
    def temporal_columns(self) -> list[str]:
        """
        Columns that go into x_temporal (all non-categorical specs).
        Order matches their position in self.specs (stable for tensor indexing).
        """
        return [s.name for s in self.specs if s.temporal]
 
    @cached_property
    def categorical_columns(self) -> list[str]:
        """
        Columns that go into x_categorical (embedded integer indices).
        """
        return [s.name for s in self.specs if not s.temporal]
 
    @cached_property 
    def snapshot_columns(self) -> list[str]:
        """
        Columns NOT accumulated across GWs (AccumulationType.NONE).
        Used by GameweekProvider for DGW snapshot separation and by
        PriorComputer for minutes-weighted averaging.
        """
        return [s.name for s in self.specs if s.is_snapshot]
    
    @cached_property 
    def per_90_columns(self) -> list[str]:
        """
        Columns that are per-90 rates (AccumulationType.PER_90).
        Used by GameweekProvider to know which columns to apply
        the per-90 calculation to after cumulative update.
        """
        return [s.name for s in self.specs if s.is_per_90]
 
    @cached_property
    def cumulative_columns(self) -> list[str]:
        """
        Columns that are accumulated (PER_90 or RAW_CUMULATIVE).
        Used by PriorComputer for rate computation.
        """
        return [s.name for s in self.specs if s.is_cumulative]
 
    @cached_property
    def raw_cumulative_columns(self) -> list[str]:
        """
        Columns that are accumulated but NOT per-90 divided.
        Typically just 'minutes' and 'featured'.
        """
        return [s.name for s in self.specs if s.accumulation == AccumulationType.RAW_CUMULATIVE]
 
    # ══════════════════════════════════════════════════════════════════
    #  Provider-Filtered Column Lists
    #  Same column lists as above, but narrowed to features that a
    #  specific DataSource actually supplies.
    # ══════════════════════════════════════════════════════════════════
    
    def columns_for(self, base_list: list[str], provider: DataSource):
        """
        Filter any column list to only those supplied by a given provider.
        """
        return [
            name for name in base_list
            if provider in self._spec_by_name[name].source_columns
            and self._spec_by_name[name].source_columns[provider] is not None
        ]

    def cumulative_columns_for(self, provider: DataSource) -> list[str]:
        """
        Cumulative output column names that exist in a given provider.
        Only includes snapshots that this provider actually supplies.
        """
        return self._columns_for(self.cumulative_columns, provider)
 
    def per_90_columns_for(self, provider: DataSource) -> list[str]:
        """
        per_90 output column names that exist in a given provider.
        Only includes snapshots that this provider actually supplies.
        """
        return self._columns_for(self.per_90_columns, provider)
 
    def snapshot_columns_for(self, provider: DataSource) -> list[str]:
        """
        Snapshot output column names that exist in a given provider.
        Only includes snapshots that this provider actually supplies.
        """
        return self._columns_for(self.snapshot_columns, provider)
 
    # ══════════════════════════════════════════════════════════════════
    #  Column Mapping Dicts
    #  Dictionaries that map between output names, cumulative names,
    #  and raw source column names.
    # ══════════════════════════════════════════════════════════════════
 
    @property
    def cumulative_map(self) -> dict[str, str]:
        """
        output_col → cum_col for all accumulated features.
        """
        return {
            s.name: s.cum_col
            for s in self.specs
            if s.cum_col is not None
        }
 
    @property
    def inv_cumulative_map(self) -> dict[str, str]:
        """
        cum_col → output_col (inverse of cum_map).
        """
        return {
            cum: out 
            for out, cum in self.cumulative_map.items()
        }
 
    def cumulative_map_for(self, provider: DataSource) -> dict[str, str]:
        """
        output_col → cum_col for all accumulated features.
        """
        return {
            s: self._spec_by_name[s].cum_col
            for s in self.cumulative_map
            if provider in self._spec_by_name[s].source_columns
            and self._spec_by_name[s].source_columns[provider] is not None
        }
 
    def inv_cumulative_map_for(self, provider: DataSource) -> dict[str, str]:
        """
        output_col → cum_col for all accumulated features.
        """
        return {
            self._spec_by_name[s].cum_col: s
            for s in self.cumulative_map
            if provider in self._spec_by_name[s].source_columns
            and self._spec_by_name[s].source_columns[provider] is not None
        }
 
    def source_map(self, provider: DataSource) -> dict[str, str]:
        """
        source_col → output_col for one data provider.
        Args:
            provider: key into FeatureSpec.source_columns, e.g. "vaastav", "fci", "opta"
        """
        result = {}
        for s in self.specs:
            if s.source_columns and provider in s.source_columns:
                source_col = s.source_columns[provider]
                if source_col is not None:
                    result[source_col] = s.name
        return result
 
    # ══════════════════════════════════════════════════════════════════
    #  Spec Lookups
    #  Access individual specs by name, index, or grouped by
    #  mode / provider.
    # ══════════════════════════════════════════════════════════════════
 
    def __getitem__(self, name: str) -> FeatureSpec:
        """
        Look up a FeatureSpec by name. Raises KeyError if not found.
        """
        try:
            return self._spec_by_name[name]
        except KeyError:
            raise KeyError(f"No feature named '{name}'. Available: {list(self._spec_by_name.keys())}")
 
    def __contains__(self, name: str) -> bool:
        return name in self._spec_by_name
 
    def index_of(self, name: str) -> int:
        """
        Position of a named feature in the tensor column ordering.
        Raises KeyError if not found.
        """
        for i, s in enumerate(self.specs):
            if s.name == name:
                return i
        raise KeyError(f"No feature named '{name}'")
 
    @property
    def specs_by_mode(self) -> dict[ScalingMode, list[FeatureSpec]]:
        return {mode: [s for s in self.specs if s.scaling_mode == mode] for mode in ScalingMode}
 
    @property
    def specs_by_provider(self) -> dict[str, list[FeatureSpec]]:
        """
        Group specs by their feature_provider field.
        """
        result: dict[str, list[FeatureSpec]] = {}
        for s in self.specs:
            result.setdefault(s.feature_provider, []).append(s)
        return result
 
    # ══════════════════════════════════════════════════════════════════
    #  Tensor Mask Builders (private)
    #  Called once at init to create boolean tensors used by the
    #  scaling and encoding layers.
    # ══════════════════════════════════════════════════════════════════
 
    def _temporal_mask(self):
        # build mask for temporal features, none temporal is inverse
        return torch.tensor([s.temporal for s in self.specs])
 
    def _build_mode_masks(self):
        # build masks for different scaling modes
        masks = {}
        for ftype in ScalingMode:
            masks[ftype] = torch.tensor(
                [s.scaling_mode == ftype for s in self.specs],
                dtype=torch.bool
            )
        return masks
 
    def _build_type_mask(self):
        # build mask by type of feature
        masks = {}
        for ftype in FeatureType:
            masks[ftype] = torch.tensor(
                [s.feature_type == ftype for s in self.specs],
                dtype=torch.bool
            )
        return masks
 
    # ══════════════════════════════════════════════════════════════════
    #  DataFrame Validation
    # ══════════════════════════════════════════════════════════════════
 
    def validate_dataframe(self, df: pd.DataFrame, context: str = "") -> None:
        """
        Check that a DataFrame has the expected output columns.
        Raises ValueError with a clear message listing missing/extra columns.
        Args:
            df: DataFrame to validate
            context: optional string for error messages (e.g. "GW5 ingester output")
        """
        expected = set(self.output_columns)
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"{context} missing columns: {sorted(missing)}")
        if extra:
            logger.warning(f"{context} has extra columns not in Features: {sorted(extra)}")
 
    # ══════════════════════════════════════════════════════════════════
    #  Serialisation
    # ══════════════════════════════════════════════════════════════════
 
    def to_dict(self):
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
    def from_dict(cls, data):
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
 
    # ══════════════════════════════════════════════════════════════════
    #  Dunder Overrides
    # ══════════════════════════════════════════════════════════════════
 
    def __len__(self):
        # return number of features
        return len(self.specs)  
