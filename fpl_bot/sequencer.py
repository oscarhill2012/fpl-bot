from __future__ import annotations
import math
import logging
import numpy as np
import pandas as pd
import torch
from torch._C import int16
from torch.utils.data import Dataset
from .ingester import Ingester, FPLSourceConfig
from .priors import PriorComputer, PriorData
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
        self.season_root = season_root
        self._ingester = Ingester(season_root, fpl_config_season, opta_config)
        self._window_size = window_size
        
        # generate list of catergorical and temporal features
        self._temporal_features = features.temporal_specs
        self._catergorical_features = features.catergorical_specs

        # index by player code for fast lookup
        if "player_code" in player_meta.columns:
            self._player_meta = player_meta.set_index("player_code")
        else:
            self._player_meta = player_meta

        # nothing ingested yet
        self._current_gameweek: int = 0
        self._prior_data = prior_data
        
        # first gameweek a player plays minutes > 0, until then we use priors
        self._first_gw: dict[str, str] = {}

        # find fixtures, (team_code, gw) -> fixture info
        self._fixture_lookup: dict[tuple[int, str], dict] = {}

        # cache data in as dict, so lookup cheaper than pandas .loc
        self.gw_cache: dict[int, dict[str, dict[str, float]]] = {}

    # --- Ingestions & State Management ---

    def _cache_gw(self, gw: int) -> "SeasonSequencer":
        """
        For dict ingester produces, convert nested df to dicts, per index.
        """
        self.gw_cache[gw] = self._ingester.player_gw_stats[gw].to_dict(orient="index")
        return self
    
    
    def ingest_range(self, gw_start: int, gw_end: int) -> "SeasonSequencer": 
        """
        Bulk-ingest a range of gameweeks. 
        Provides priors based on range if have None.
        """
        # reset ingester to ensure clean cumulative states
        self._ingester.reset()

        self._ingester.ingest(gw_start, gw_end)
        self._first_gw = self._ingester.first_gw
        self._current_gameweek = gw_end

        # add new data to gw_cache
        for gw in range(gw_start, gw_end-1):
            self._cache_gw(gw)

        if self._prior_data is None:
            self._prior_data = PriorData.from_data(
                self._ingester.player_gw_stats,
                self._ingester.player_cumulative_stats,
                self._player_meta,
                self._ingester.cum_rev_map,
                min_mins=450.0
            )
            self._prior_data.to_json(self.season_root)

        return self

    def step(self, teams_df: pd.DataFrame | None = None) -> "SeasonSequencer":
        """
        Advances the sequencer by one Gameweek.
        Uses Ingesters append_gw().
        Optionally can refresh team strength and elo.
        """
        # require priors to step
        if self._prior_data is None:
            raise RuntimeError("Please provide priors or ingest_range() to step()")
        
        self._current_gameweek += 1

        self._ingester.append_gw(self._current_gameweek)
        self._first_gw = self._ingester.first_gw

        self._cache_gw(self._current_gameweek)

        if not teams_df is None:
            self._update_team_strength(teams_df)
        
        return self

    def _update_team_strength(self, teams_df: pd.DataFrame) -> "SeasonSequencer":
        """
        Update team stregth and elo.
        """
        self._team_strength = self._parse_team_strength(teams_df)
        logger.info("Updated team strength")
        
        return self

    # --- Sequence Construction ---

    def _build_window(self, target_gw: int) -> list[int]:
        """
        Builds list of gameweeks in window.
        """
        # window must always start at seq_start > 0
        seq_start = max(1, target_gw - self._window_size)

        return list(range(seq_start, target_gw-1))

    def _get_prior(self, player_team_id: str, team_code: str, position: str) -> dict[str, float]:
        """
        Lookup for prior row, given player_team_id and position
        """
        if self._prior_data is None:
            raise RuntimeError("No prior data available, call ingeste_range() first.")

        pos_team = "_".join([position, team_code])       # NOTE: order is convention

        # lookup player in prior hierachy
        if player_team_id in self._prior_data.individual:
            prior_row = self._prior_data.individual[player_team_id]

        elif pos_team in self._prior_data.position_team:            
            prior_row = self._prior_data.position_team[pos_team]
            prior_row["minutes"] = self._minutes_lookup(prior_row)
        elif position in self._prior_data.position:
            prior_row = self._prior_data.position[position]
            prior_row["minutes"] = self._minutes_lookup(prior_row)
        else:
            prior_row = self._prior_data.league["league"]
            prior_row["minutes"] = self._minutes_lookup(prior_row)

        # add row to denote prior, "data_age" is sentinal 
        prior_row["is_prior"], prior_row["data_age"] = 1.0, 0.0

        return prior_row 

    @staticmethod
    def _minutes_lookup(prior_row: dict[str, float]) -> float:
        """
        Lookup table, generated per step.
        Scales minutes of player with no data, 
        to ensure priors is more realistic 
        """
        return 0.0

    def _get_real_row(self, player_team_code: str, gw: int) -> dict[str, float]:
        """
        Gets a players stats, for given gw.
        """
        # player lookup in cache
        real_row = self.gw_cache[gw][player_team_code]

        # add row to denote prior, "data_age" is sentinal 
        real_row["is_prior"], real_row["data_age"] = 0.0, 0.0
        return real_row

    def _verify_consistent_features(self, list1: list[str], list2: list[str]):
        """
        Optional validation fuction, ensure both lists contain same features.
        """
        if list1 != list2:
            raise RuntimeError("Inconsistent features between prior and current data")
        return self
    
    def build_player_window(self, player_team_id: str, window: list[int], inference: bool = False) -> dict:
        """
        constructs fixed length (_window_size) sequence up to target_gw,
        for each timestep decides if to use priors or real data
        """
        position = self.player_meta.loc[player_team_id, "position"]
        team_code = self.player_meta.loc[player_team_id, "team_code"]   
        first_gw = self._first_gw[player_team_id]
        gw_window: list[int] = []

        # filter any gameweeks before, players first appearance
        # if player hasn't appeared yet first gw = 39 so window = []
        gw_window[:] = [i for i in window if i > first_gw]

        # window must always be len self._window_size, 0 in window => prior row
        n_pad = self._window_size - len(window)
        gw_window[:0] = [0] * n_pad

        if n_pad != 0:
            prior_row = self._get_prior(player_team_id, team_code, position) 
            self._verify_consistent_features(list(prior_row.keys()), self._temporal_features + ["is_prior", "data_age"])

        data_list = []
        for i, gw in enumerate(gw_window):
            if gw == 0:
                prior_row["data_age"] = self.window_size - i
                data_list.append(list(prior_row.values()))
            else:
                real_row = self._get_real_row(player_team_id, gw)
                data_list.append(list(real_row.values()))
                self._verify_consistent_features(list(real_row.keys()), self._temporal_features + ["is_prior", "data_age"])

        # conver to 2D numpy array, shape [n_timesteps, n_features]
        x_temporal = np.array(data_list, dtype=np.float32)  
            
        

