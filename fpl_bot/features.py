import torch
import logging
from enum import Enum  
import math
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


class FeatureSpec:

    def __init__(
        self,
        name: str,
        feature_type: FeatureType,
        scaling_mode: ScalingMode,
        temporal: bool = True,
        presence_check: bool = False,
        categories: list | None = None,
        embedding_dim: int | None = None,
        period: int | None = None,
        max_value: float | None = None,
        min_value: float = 0,        
        scaling_params: list | None = None,
    ):
        self.name = name                                # feature name
        self.feature_type = feature_type                # data type of feature
        self.scaling_mode = scaling_mode                # how feature will be scaled
        self.temporal = temporal                        # does feature vary with time
        self.presence_check = presence_check            # if true, any sample with feature value = min value is missing from data NOT 0
        self.categories = categories                    # if catergorical type, what catergories
        self.embedding_dim = embedding_dim              # if embedded, how many dims
        self.max_value = max_value                      # max value of feature, for bounded
        self.min_value = min_value                      # min value of feature, for bounded and presence check
        self.period = period                            # period of cyclic feature
        self.scaling_params = scaling_params if scaling_params is not None else [None, None]                 
                                                        # parameters fit from scaling 


class Features:

    def __init__(self, specs: list[FeatureSpec], eps=1e-8):
        self.specs = specs
        self.eps = eps
        self._validate()
        self.scaling_masks = self._build_mode_masks()       # builds tensor, a mask for each scaling mode, shape = [n_features]
        self.type_masks = self._build_type_mask()           # builds tensor, a mask for each feature type, shape = [ n_features]
        self.temporal_mask = self._temporal_mask()          # builds tensor, a temporal mask, True is tempooral, shape = [n_features]
    
    # --- Validation ---

    def _validate(self):
        # validate input
        # ensure no duplicates
        names = [s.name for s in self.specs]
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
            # catergorical feature requires catergories
            if s.feature_type == FeatureType.CATEGORICAL:
                if s.categories is None:
                    raise ValueError(f"{s.name} requires catergories")
            # a feature that implies data was present (i.e 0 means bad performance not missing) requires threshold
            if s.presence_check == True:
                if s.min_value is None:
                    raise ValueError(f"{s.name} requires min_value to be presence check")

    # --- Propterties ---

    @property
    def names(self):
        return [s.name for s in self.specs]

    @property
    def specs_by_mode(self):
        return {mode: [s for s in self.specs if s.scaling_mode == mode] for mode in ScalingMode}

    # --- Helper Functions ---

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

    # --- Public Functions and Methods ---

    def __len__(self):
        # return number of features
        return len(self.specs)  

    def to_dict(self):
        return [
            {
                "name": s.name,
                "feature_type": s.feature_type.value,
                "scaling_mode": s.scaling_mode.value,
                "scaling_params": s.scaling_params,
                "temporal": s.temporal,
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
                    feature_type=FeatureType(d["feature_type"]),
                    scaling_mode=ScalingMode(d["scaling_mode"]),
                    scaling_params=d["scaling_params"],
                    temporal=d["temporal"],
                    presence_check=d["presence_check"],
                    categories=d["categories"],
                    embedding_dim=d["embedding_dim"],
                    period=d["period"],
                    max_value=d["max_value"],
                    min_value=d["min_value"],
                )
            )
        return cls(specs)   

