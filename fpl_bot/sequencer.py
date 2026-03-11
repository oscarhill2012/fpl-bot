import math
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .ingester import Ingester, FPLSourceConfig
from .priors import compute_priors, PriorData
from .features import Features, FeatureSpec, FeatureType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Position strings → integer indices for categorical embedding.
# 0 is reserved as a padding index (used when position is unknown or
# for the padding_idx argument of nn.Embedding, which zeros out the
# gradient for that index so the model doesn't learn from empty slots).
_POSITION_TO_IDX = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}


class SeasonSequencer:
    """
    Stateful orchestrator for one season of FPL data.

    Owns the Ingester and PriorData internally. 
    Exposes sequence building and dataset creation to the caller. 
    Advancing through gameweeks is done via step(),
    which ingests one new GW and updates all internal derived state (carry-forward, firstappearances).

    Typical lifecycle:
        seq = SeasonSequencer(...)
        seq.ingest_range(1, 30)       # bulk load GW1-30
        ds = seq.dataset(1, 30)       # torch Dataset for training
        seq.step()                    # ingest GW31
        preds = seq.predict_batch(32) # predict GW32
    """
    def __init__(
        self,
        season_root: str,
        fpl_config_season: FPLSourceConfig,
        opta_config: FPLSourceConfig,
        player_meta: pd.DataFrame,
        fixture_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        features: Features,
        window_size: int=8,
        prior_data: PriorData | None = None,
    ):
        ### --- Internal Components ---
        self._ingester = Ingester(season_root, fpl_config_season, opta_config)
        self._features = features
        self._window_size = window_size
        
        # index by player code for fast lookup
        if "player_code" in player_meta.columns:
            self._player_meta = player_meta.set_index("player_code")
        else:
            self.player_meta = player_meta

        # nothing ingested yet
        self._current_gameweek: int = 0
        self._prior_data = prior_data
        self._carried: dict[str, pd.DataFrame] = {}
        
        # first gameweek a player plays minutes > 0, until then we use priors
        self._first_gw: dict[int, str] = {}

        # find fixtures, (team_code, gw) -> fixture info
        self._fixture_lookup: dict[tuple[int, str], dict] = {}


    # --- Ingestions & State Management ---

    def ingest_range(self, gw_start: int, gw_end: int) -> "SeasonSequencer": 
        """
        Bulk-ingest a range of gameweeks. 
        Resets ingester first to ensure clean cumulative state.
        """
        self._ingester.reset()

        
        