import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
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
    "total_shots":              "shots_per_90",
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
    "xgot_faced":               "xgot_faced_per_90",
    "goals_prevented":          "goals_prevented_per_90",
    "sweeper_actions":          "sweeper_actions_per_90",
    "high_claim":               "high_claim_per_90",
    "gk_accurate_passes":       "gk_accurate_passes_per_90",
    "gk_accurate_long_balls":   "gk_accurate_long_balls_per_90",
}

vaastav_map = {
    "minutes":                      "minutes",
    "starts":                       "starts",
    "value":                        "price",
    "total_points":                 "total_points",
    "transfers_in":                 "transfers_in",
    "transfers_out":                "transfers_out",
    "goals_scored":                 "goals_per_90",
    "assists":                      "assists_per_90",
    "clean_sheets":                 "clean_sheets_per_90",
    "goals_conceded":               "goals_conceded_per_90",
    "yellow_cards":                 "yellow_cards_per_90",
    "saves":                        "saves_per_90",
    "bonus":                        "bonus_per_90",
    "bps":                          "bps_per_90",
    "influence":                    "influence_per_90",
    "creativity":                   "creativity_per_90",
    "threat":                       "threat_per_90",
    "ict_index":                    "ict_index_per_90",
    "expected_goals":               "expected_goals_per_90",
    "expected_assists":             "expected_assists_per_90",
    "expected_goal_involvements":   "expected_goal_involvements_per_90",
    "expected_goals_conceded":      "expected_goals_conceded_per_90",
}

fci_map = {
    "minutes":                      "minutes",
    "starts":                       "starts",
    "now_cost":                     "price",
    "total_points":                 "points",
    "transfers_in":                 "transfers_in",
    "transfers_out":                "transfers_out",
    "goals_scored":                 "goals_per_90",
    "assists":                      "assists_per_90",
    "clean_sheets":                 "clean_sheets_per_90",
    "goals_conceded":               "goals_conceded_per_90",
    "yellow_cards":                 "yellow_cards_per_90",
    "saves":                        "saves_per_90",
    "bonus":                        "bonus_per_90",
    "bps":                          "bps_per_90",
    "influence":                    "influence_per_90",
    "creativity":                   "creativity_per_90",
    "threat":                       "threat_per_90",
    "ict_index":                    "ict_index_per_90",
    "expected_goals":               "expected_goals_per_90",
    "expected_assists":             "expected_assists_per_90",
    "expected_goal_involvements":   "expected_goal_involvements_per_90",
    "expected_goals_conceded":      "expected_goals_conceded_per_90",
}

vaastav_transform = {
    "price": lambda x: x/10,
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
    id_map: pd.DataFrame              # dataframe that contains id inrelation to player code
    stacked: bool                     # one big file vs per-GW files
    denotes_epl: dict[str, str]       # verifies data from epl match: {indentifying feature: identifying string}
    other_games: bool                 # if dataset contains non-EPL matches, is True
    gw_col: str | None                # which col has the GW number
    gw_path: str | None               # path pattern for per-GW files
    transform: dict[str, callable]    # transform feature


class dfProvider:
    """
    loads per-GW FPL API stats from any source.
    Cummulative totals are stored as internal state, 
    please use reset() to clean
    """
    def __init__(
        self, 
        config: FPLSourceConfig,
        season_root: str,
    ):
        self.cfg = config
        self.season_root = season_root
        self._cum = None
        self._cum_map = {}

        # check if stacked df, one initial load saves time
        if self.cfg.stacked is True:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

        # generate columns of cumulative data 
        for cols in self.cfg.col_map.values():
            self._cum_map[cols] = "cum_" + cols.replace("_per_90", "")
    
    def reset(self):
        self._cum = None
        return self

    def load_gameweek(
        self, 
        gw: int, 
        ) -> pd.DataFrame:
        """
        - loads a gameweek, from stacked or from file
        - processes
        - calculates any per_90 feature if config sets output to contain "per_90"
        - stores cumulative tally privately and outputs df
        """
        # load df
        if self.cfg.stacked is True:
            df = self._stacked_data[self._stacked_data[self.cfg.gw_col] == gw]
        else:    
            df = pd.read_csv(self.season_root + f"/GW{gw}/playermatchstats.csv")
        
        # checks all features of col_map are in dataset
        missing = set(self.cfg.col_map.keys()) - set(df.columns)
        if missing:
            logger.warning(f"GW{gw}: missing source columns: {missing}")

        # if df contains none epl matches, remove them
        if self.cfg.other_games is True:
            col, pattern = next(iter(self.cfg.denotes_epl.items()))
            df = df[df[col].str.contains(pattern)]

        # checks if data set empty
        if df.empty:
            logger.warning(f"GW{gw}: no data left after filtering for EPL")

        # load output columns to list
        output_cols = list(self.cfg.col_map.values())

        # process df
        df = (
            df
            .rename(columns=self.cfg.player_id | self.cfg.col_map)                      # enforce naming convention         
            .groupby("player_id", as_index=False).sum(numeric_only=True)                # sums incase of dgw
            .merge(self.cfg.id_map, on="player_id", how="right")                        # add all players (including those who didnt play),
            .fillna(0)                                                                  # setting all stats 0 if they didn't feature
            .set_index("player_code")                                                   # index by player code
            .filter(items=output_cols)                                                  # select only output columns
            .apply(**{col: func(df[col]) for col, func in self.cfg.transform.items()})  # apply transforms
        )

        # snapshot previous points
        prev_points = self._cum["cum_points"].copy() if self._cum is not None else 0 

        # populate cumulative df if none exists, else add to existing df
        if self._cum is None:
            self._cum = df[output_cols].rename(columns=self._cum_map).copy()
        else:
            self._cum = self._cum.add(df[output_cols].rename(columns=self._cum_map), fill_value = 0)

        # if dataset contains points, it needs to be differenced (we want points from given game)
        if "points" in df.columns:
            df["points"] = self._cum["cum_points"] - prev_points
            df = df.drop("points", axis=1)


        # per_90_ify and for rows of df where minutes != 0, this implies cum =! 0 by definition
        per_90_features = [v for v in output_cols if "per_90" in v]
        mask = df["minutes"] != 0
        df.loc[mask, per_90_features] = self._cum.loc[mask, per_90_features].div(self._cum.loc[mask, "cum_minutes"] / 90, axis=0)
      
        return df

        
class Ingester:
    """
    Ingests data from a source depending on the provided config,
    Outputs both snapshot and cumulative data, output stored in dict.
    Convention; ingester only to be used for one season at a time,
                output dict is {int, DataFrame},
                with int is season start year and gw, i.e 2438 (24/25 season, gw 38)
    WARNING: both opta and fpl_api contain minutes that slightly differ,
             here we use opta minutes for opta per 90 and hope it makes no difference!
             - when tested differences were at most 5% over season so should be fine
    """
    def __init__(
        self,
        season_root: str,
        fpl_config: FPLSourceConfig,
        opta_config: FPLSourceConfig,
    ):
        self.season_root = season_root
        self.fpl_provider = dfProvider(fpl_config, season_root)
        self.opta_provider = dfProvider(opta_config, season_root)
        self.output_dict = {}
        self.cumulative_dict = {}

    """
    Private Functions
    """

    def _merge_dfs(self, fpl: pd.DataFrame, opta: pd.DataFrame) -> pd.DataFrame:
        # any final processing for fpl and opta datasets
        fpl = (
            fpl
        )
        opta = (
            opta
        )
        return pd.concat([fpl, opta], axis=1)

    """
    Public Functions
    """

    def reset(self):
        logger.info("FPL and Opta cumulative tallies have been reset.")
        self.fpl_provider.reset()
        self.opta_provider.reset()
        return self       

    def ingest(self, gameweek_start: int, gameweek_end: int) -> dict[str, pd.DataFrame]:
        """
        Ingests bulk data from a gameweek range
        """
        # warn user if running ingest() without reset
        if not self.cumulative_dict:
            logger.warning("Cumulative tally has not been reset, proceed with caution.")

        # load opta and fpl api stats
        for gw in range(gameweek_start, gameweek_end+1):
            # drop minutes from opta data
            gw_combined = self._merge_dfs(
                self.fpl_provider.load_gameweek(gw), 
                self.opta_provider.load_gameweek(gw).drop(columns="minutes", errors="ignore"),
            )
            # drop cumulative points as this was only used for differencing to get points
            gw_cum_combined = self._merge_dfs(
                self.fpl_provider._cum.drop(columns="cum_minutes", errors="ignore"), 
                self.opta_provider._cum.drop(columns="cum_minutes", errors="ignore"),
            )
            self.output_dict[gw] = gw_combined
            self.cumulative_dict[gw] = gw_cum_combined

        return self.output_dict, self.cumulative_dict

    def append_gw(self, gw: int) -> dict[str, pd.DataFrame]:
        """
        Ingests a given gameweek
        WARNING: for functionality append_gw() never resets cummulative data,
                 please explicitly call reset() if required.
        """
        # load opta and fpl api stats
        # drop minutes from opta data
        gw_combined = self._merge_dfs(
            self.fpl_provider.load_gameweek(gw), 
            self.opta_provider.load_gameweek(gw).drop(columns="minutes", errors="ignore"),
        )
        # drop cumulative points as this was only used for differencing to get points
        gw_cum_combined = self._merge_dfs(
            self.fpl_provider._cum.drop(columns="cum_minutes", errors="ignore"), 
            self.opta_provider._cum.drop(columns="cum_minutes", errors="ignore"),
        )
        self.output_dict[gw] = gw_combined
        self.cumulative_dict[gw] = gw_cum_combined

        return self.output_dict, self.cumulative_dict