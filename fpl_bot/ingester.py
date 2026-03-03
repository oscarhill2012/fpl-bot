import torch 
import pandas as pd
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable
import math 
from .features import Features, FeatureSpec, FeatureType
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


valid_columns_opta = {
    'player_id':              'FPL player ID',
    'match_id':               'match identifier string',
    'minutes_played':         'minutes on pitch',
    'start_min':              'timing field (unreliable as entry time)',
    'finish_min':             'end-of-match minute marker',
    'goals':                  'goals scored',
    'assists':                'assists',
    'xg':                     'expected goals',
    'xa':                     'expected assists',
    'xgot':                   'expected goals on target',
    'total_shots':            'total shots',
    'shots_on_target':        'shots on target',
    'big_chances_missed':     'big chances missed',
    'accurate_passes':        'accurate passes',
    'chances_created':        'chances created',
    'final_third_passes':     'passes into final third',
    'accurate_crosses':       'accurate crosses',
    'accurate_long_balls':    'accurate long balls',
    'touches':                'total touches',
    'touches_opposition_box': 'touches in opposition box',
    'successful_dribbles':    'successful dribbles',
    'tackles':                'total tackles',
    'interceptions':          'interceptions',
    'recoveries':             'ball recoveries',
    'blocks':                 'blocks',
    'clearances':             'clearances',
    'headed_clearances':      'headed clearances',
    'duels_won':              'duels won',
    'duels_lost':             'duels lost',
    'ground_duels_won':       'ground duels won',
    'aerial_duels_won':       'aerial duels won',
    'dribbled_past':          'times dribbled past',
    'was_fouled':             'times fouled',
    'fouls_committed':        'fouls committed',
    'saves':                  'saves (GK only)',
    'goals_conceded':         'goals conceded (GK only)',
    'xgot_faced':             'xGOT faced (GK only)',
    'goals_prevented':        'goals prevented (GK only)',
    'sweeper_actions':        'sweeper actions (GK only)',
    'gk_accurate_passes':     'GK accurate passes (GK only)',
    'gk_accurate_long_balls': 'GK accurate long balls (GK only)',
    'high_claim':             'high claims (GK only)',
    'team_goals_conceded':    'team goals conceded (bugged for subs)',
}

opta_map = {
    # Unchanged
    "minutes_played":           "minutes",

    # Attack
    "goals":                    "goals_per_90",
    "assists":                  "assists_per_90",
    "total_shots":              "shots_per_90",
    "xg":                       "xg_per_90",
    "xa":                       "xa_per_90",
    "xgot":                     "xgot_per_90",
    "shots_on_target":          "shots_on_target_per_90",
    "big_chances_missed":       "big_chances_missed_per_90",
    "chances_created":          "chances_created_per_90",
    "touches_opposition_box":   "touches_opposition_box_per_90",

    # Passing
    "accurate_passes":          "accurate_passes_per_90",
    "final_third_passes":       "final_third_passes_per_90",
    "accurate_crosses":         "accurate_crosses_per_90",
    "accurate_long_balls":      "accurate_long_balls_per_90",

    # Possession
    "touches":                  "touches_per_90",
    "successful_dribbles":      "successful_dribbles_per_90",

    # Defence
    "tackles":                  "tackles_per_90",
    "interceptions":            "interceptions_per_90",
    "recoveries":               "recoveries_per_90",
    "blocks":                   "blocks_per_90",
    "clearances":               "clearances_per_90",
    "headed_clearances":        "headed_clearances_per_90",
    "dribbled_past":            "dribbled_past_per_90",

    # Duels
    "duels_won":                "duels_won_per_90",
    "duels_lost":               "duels_lost_per_90",
    "ground_duels_won":         "ground_duels_won_per_90",
    "aerial_duels_won":         "aerial_duels_won_per_90",

    # Discipline / Other
    "was_fouled":               "was_fouled_per_90",
    "fouls_committed":          "fouls_committed_per_90",

    # GK
    "saves":                    "saves_per_90",
    "goals_conceded":           "goals_conceded_per_90",
    "xgot_faced":               "xgot_faced_per_90",
    "goals_prevented":          "goals_prevented_per_90",
    "sweeper_actions":          "sweeper_actions_per_90",
    "high_claim":               "high_claim_per_90",
    "gk_accurate_passes":       "gk_accurate_passes_per_90",
    "gk_accurate_long_balls":   "gk_accurate_long_balls_per_90",
}

@dataclass
class FPLSourceConfig:
    """
    Encodes distinct natures of data sets so all can be treated equally,
    all output columns required must be in map (bar defaults), even if no transform,
    any defaults will be added to map automatically
    """
    col_map: dict[str, str]           # source_col → output columns (should contain all columns even if no change)
    player_id: dict[str, str]         # maps player_id tag in dataset to "player_id"
    stacked: bool                     # one big file vs per-GW files
    gw_col: str | None                # which col has the GW number
    gw_path: str | None               # path pattern for per-GW files
    defaults: dict[str, float]        # missing cols → default values. NOTE: this stomps existing columns, intended feature

class FPL_API_Provider:
    """
    loads per-GW FPL API stats from any source.
    """
    def __init__(
        self, 
        config: FPLSourceConfig,
        season_root: str,
        id_map: pd.DataFrame,
    ):
        self.cfg = config
        self.season_root = season_root
        self.id_map = id_map

        if self.cfg.stacked is True:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

    def load_gameweek(
        self, 
        gw: int,
    ) -> pd.DataFrame:

        # load df, from mememory if stacked and from file if not
        if self.cfg.stacked is True:
            df = self._stacked_data[self._stacked_data[self.cfg.gw_col] == gw]
        else:
            df = pd.read_csv(self.season_root + self.cfg.gw_path + f"GW{gw}")

        # ensure that our output will contain any added defaults
        output_cols = list(self.cfg.col_map.values()) + list(self.cfg.defaults.keys())    

        # process df      
        df = (
            df
            .rename(columns=self.cfg.player_id | self.cfg.col_map)  # renames columns
            .assign(**self.cfg.defaults)                            # add default columns if missing
            .merge(self.id_map, on="player_id", how="left")         # currently maps, adding NO rows    
            .fillna(0)                                              # players missing from data, but in map, get 0s                 
            .set_index("player_code")                               # set player_code as index
            .filter(items=output_cols, axis=1)                      # select only output columns
        )
        return df

        

class Ingester:

    def __init__(
        self,
        id_map: pd.DataFrame,
        season_root: str,
        opta_cfg: FPLSourceConfig,
        fpl_provider: FPL_API_Provider,
    ):
        self.id_map = id_map
        self.season_root = season_root
        self.opta_cfg = opta_cfg
        self.fpl_provider = fpl_provider

    """
    - loads one GW opta stats from FPL-Core-Insights repo,
    - since per_90 stats ae (cumulative_stat / cumulative_minutes) * 90,
      we will internally accumulate stats and minutes, 
      this accumulative store will need to contain all players in id_map.
    - this source also includes europe, cup games and DGW, we isolate only EPL matches
    """
    def _load_pms_gameweek(
        self, 
        gw: int, 
        prev_cum: pd.DataFrame | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        WARNING: If previous cummulative data exists, please input or tally reset.
        this is required defaults and transforms for cfg setup later
        defaults {
            "_featured": 1,
        }
        """
        # add current game week to defaults and adds defaults to output columns
        defaults = {**self.opta_cfg.defaults, "gw": gw}
        output_cols = list(self.opta_cfg.col_map.values()) + list(defaults.keys())

        # load df
        df = pd.read_csv(self.season_root + f"/GW{gw}/playermatchstats.csv")

        # process df
        df = (
            df
            .rename(columns=self.opta_cfg.player_id | self.opta_cfg.col_map)    # enforce naming convention
            .pipe(lambda d: d[["prem" in x for x in d["match_id"]]])            # extracts only EPL games         
            .groupby("player_id", as_index=False).sum(numeric_only=True)        # sums incase of dgw
            .assign(**defaults)                                                 # adds defaults
            .merge(self.id_map, on="player_id", how="right")                    # add all players (including those who didnt play), 
            .fillna(0)                                                          # setting all stats 0 if they didn't feature
            .set_index("player_code")                                           # index by player code
            .filter(items=output_cols)                                          # select only output columns
        )

        # populate cumulative df if none exists, else add to existing df
        if prev_cum is None:
            cum = df[self.opta_cfg.col_map.values()].copy()     
        else:
            cum = prev_cum.add(df[self.opta_cfg.col_map.values()], fill_value = 0)

        # per_90_ify all features and for rows of df where minutes != 0, this implies cum =! 0 by its definition
        per_90_features = [v for v in opta_map.values() if "per_90" in v]
        mask = df["minutes"] != 0
        df.loc[mask, per_90_features] = cum.loc[mask, per_90_features].div(cum.loc[mask, "minutes"] / 90, axis=0)

        # adds defaults
                                                  
        return df, cum
