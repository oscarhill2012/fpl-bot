import pandas as pd
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

fci_map = {
    "minutes":                      "minutes",
    "starts":                       "starts",
    "now_cost":                     "price",
    "event_points":                 "points",
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
    col_map: dict[str, str]                     # source_col → output columns (should contain all columns even if no change)
    player_id: dict[str, str]                   # maps player_id tag in dataset to "player_id"
    id_map: pd.DataFrame                        # dataframe that contains id inrelation to player code
    snapshot_cols: list[str] | None             # list of features that shouldn't be accumulated
    stacked: bool                               # one big file vs per-GW files
    denotes_epl: dict[str, str]                 # verifies data from epl match: {indentifying feature: identifying string}
    other_games: bool                           # if dataset contains non-EPL matches, is True
    gw_col: str | None                          # which col has the GW number
    gw_path: str | None                         # path pattern for per-GW files
    transform: dict[str, callable[[str], any]]  # transform feature


class GameweekProvider:
    """
    Loads per-GW stats from Vaastav/FPL (FPL API) or FPL-Core-Insights (FPL API and Opta).
    Calculates per_90 stats for a given gameweek from cumulative state, which allows calls over multiple gameweeks.
    Reset() used for internal state.
    Outputs a mixture of per_90 and snapshot statistics. 
    """
    def __init__(
        self, 
        config: FPLSourceConfig,
        season_root: str,
    ):
        self.cfg = config
        self.season_root = season_root
        self.cumulative_df = None

        # check if stacked df, one initial load saves time
        if self.cfg.stacked:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

        # generate columns of cumulative data
        # map between per_90 and cumulative, and visa versa
        self._cum_map = self._generate_cumulative_map()
        self._cum_rev = self._invert_dict(self._cum_map)
        # minutes should be in cumulative map, but isnt a per_90 so have to add manually
        self._cum_map["minutes"] = "cum_minutes"
        # initialise internal state to store points of previous week
        self._prev_points = None
    
    # --- Helper Functions ----

    def _generate_cumulative_map(self) -> dict[str, str]:
        """
        any feature that becomes per_90, needs to be accumulated to state,
        as per_90 = (cumulative total / cumulative minutes) * 90.
        This function generate dict to map per_90 feature names to cumulative "cum_feature".
        """
        return {
            per_90: "cum_" + per_90.replace("_per_90", "")
            for per_90 in self.cfg.col_map.values() 
            if "per_90" in per_90   
        }

    def _invert_dict(self, dictionary: dict[any, any]) -> dict[any, any]:
        """
        Swaps keys and values of a dict
        """
        return {
            value: key
            for key, value in dictionary.items()
        }
    
    def _match_identifier(self, df: pd.DataFrame, identifier: dict[str, str]) -> pd.DataFrame:
        """
        Selects rows from data frame that, satisfy;
        in column (identifier key) has string containing identifier value
        """
        for col, string in identifier.items():
            # regex ensures string can contain special characters
            df = df[df[col].str.contains(string, regex=False)]
        
        return df

    def _force_numeric_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Forces all cols in df to numeric.
        Returns:- df of same shape as input
        """
        # ensure we only use columns that exist in df
        cols_to_coerce = [c for c in cols if c in df.columns]
        # make columns numeric, if can't make 0.0
        df[cols_to_coerce] = df[cols_to_coerce].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return df

    def _separate_snapshot_cols(self, df: pd.DataFrame, identity_col: str, cum_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Seperate any already snapshot features from df,
        Expect more than one row per indentity.
        Confusingly, vaastav uses two columns for DGW and FCI combines,
        Vaastav: "kickoff time" gives dates of both games
        Returns:- df without snapshot columns
                - snapshot columns indexed by identity
        """
        if "kickoff_time" in df.columns:
            # kickoff time is iso string, to_datetime handles that
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
            # most recent game is now last row per id 
            df = df.sort_values("kickoff_time")
        
        # last row is most up-to-date
        snapshot_features = df.groupby(identity_col)[cum_cols].last()
        df = df.drop(columns=cum_cols)
        
        return df.reset_index(), snapshot_features

    def _update_cumulative_frame(self, df: pd.DataFrame, output_cols: list[str]):
        """
        Updates cumulative state dataframe, 
        with df no cumulative tally or by addition if is,
        points are special feature and are directly copied in not added, as recieve cumulative tally.
        Returns:- updated cumulative dataframe state
        """
        # populate cumulative df if none exists, else add to existing df
        # cummulative df has columns given by _cum_map
        if self.cumulative_df is None:
            self.cumulative_df = df[output_cols].rename(columns=self._cum_map).copy()
        else:
            # extract last total points to difference
            self.cumulative_df = self.cumulative_df.add(df[output_cols].rename(columns=self._cum_map), fill_value = 0.0)

        return self

    def _calculate_per_90(self, df: pd.DataFrame, mask: pd.Series, features: list[str]) -> pd.DataFrame:
        """
        Calculate per_90 rates for features if mask is True for row.
        Returns:- dataframe of same shape as input
        """
        df.loc[mask, features] = (
            self.cumulative_df.loc[mask, self._cum_rev.keys()]
            .rename(columns=self._cum_rev)
            .div(self.cumulative_df.loc[mask, "cum_minutes"] / 90, axis=0)
        )
        return df
    
    # --- Public Functions ---

    def reset(self):
        self.cumulative_df = None
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
        if self.cfg.stacked:
            # returns a view, so copy
            df = self._stacked_data[self._stacked_data[self.cfg.gw_col] == gw].copy()
        else:    
            df = pd.read_csv(self.season_root + f"/GW{gw}/playermatchstats.csv")
        
        # guard: checks all features of col_map are in dataset
        missing = set(self.cfg.col_map.keys()) - set(df.columns)
        if missing:
            raise KeyError(f"GW{gw}: missing source columns: {missing}")

        if self.cfg.other_games is True:
            df = self._match_identifier(df, self.cfg.denotes_epl)

        # if empty, denotes_epl is likely wrong, warn user of this
        if df.empty:
            logger.warning(f"GW{gw}: no data left after filtering for EPL")

        # load output columns to list
        output_cols = list(self.cfg.col_map.values())
        
        # rename to output column names
        df = df.rename(columns=self.cfg.player_id | self.cfg.col_map)

        # guard: coerce all output columns to numeric (real data may have string-typed floats)
        df = self._force_numeric_cols(df, output_cols)

        # does df contain snapshot features
        if self.cfg.snapshot_cols:     
            df, snapshot_features = self._separate_snapshot_cols(df, "player_id", self.cfg.snapshot_cols)

        # aggregate (sums DGW appearances)
        df = df.groupby("player_id").sum(numeric_only=True)

        # add back snapshot_features now we have summed
        if self.cfg.snapshot_cols:
            df[self.cfg.snapshot_cols] = snapshot_features

        # process df
        df = (
            df  
            .reset_index()                                          # reset index from .groupby      
            .merge(self.cfg.id_map, on="player_id", how="right")    # add all players (including those who didnt play),
            .fillna(0.0)                                            # setting all stats 0 if they didn't feature
            .set_index("player_code")                               # index by player code
            .filter(items=output_cols)                              # select only output columns
            .astype(float)                                          # cast all columns as float, player_code safe as index
        )

        # apply transforms
        if self.cfg.transform:
            df = df.assign(**{
                col: func(df[col])
                for col, func in self.cfg.transform.items() if col in df.columns
            })

        self._update_cumulative_frame(df, output_cols)

        # per_90_ify and for rows of df where minutes != 0, this implies cum =! 0 by definition
        per_90_features = [v for v in output_cols if "per_90" in v]
        # only per_90ify if minutes not 0, else div by 0
        mask = df["minutes"] > 0.0
        
        if per_90_features and mask.any():
            df = self._calculate_per_90(df, mask, per_90_features)

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
        self.fpl_provider = GameweekProvider(fpl_config, season_root)
        self.opta_provider = GameweekProvider(opta_config, season_root)
        self.player_gw_stats: dict[int, pd.DataFrame] = {}
        self.player_cumulative_stats: dict[int, pd.DataFrame] = {}
        self.first_gw: dict[int, int] = {}

    # --- Helper Functions ---

    def _merge_dfs(self, fpl: pd.DataFrame, opta: pd.DataFrame) -> pd.DataFrame:
        # remove any columns from opta that overlap with fpl
        # NOTE: we prioritise fpl data, when tested minutes differed to 5% in opta but else consistent
        overlap = fpl.columns.intersection(opta.columns)
        opta = opta.drop(columns=overlap, errors="ignore")
        return pd.concat([fpl, opta], axis=1)

    def _update_first_gw(self, df: pd.DataFrame, gw: int) -> "Ingester":
        """
        Stores first gameweek a player appears in, to dict.
        """
        # isolate players that played this gameweek
        played = df.index[df["minutes"] > 0]

        # pandas search in C so quicker than if statement
        new_players = played.difference(self.first_gw.keys())

        self.first_gw.update({pc: gw for pc in new_players})
        return self

    def _process_gw(self, gw: int):
        # uses instances of GameweekProvider to load gw
        gw_combined = self._merge_dfs(
            self.fpl_provider.load_gameweek(gw), 
            self.opta_provider.load_gameweek(gw),
        )
        # extracts cumulative tally attribute from GameweekProvider instance
        gw_cum_combined = self._merge_dfs(
            self.fpl_provider.cumulative_df, 
            self.opta_provider.cumulative_df,
        )
        self._update_first_gw(gw_combined[gw], gw)
        return gw_combined, gw_cum_combined

    # --- Public Functions ---

    def reset(self):
        logger.info("FPL and Opta cumulative tallies have been reset.")
        self.fpl_provider.reset()
        self.opta_provider.reset()
        self.first_gw = {}
        return self       

    def ingest(self, gameweek_start: int, gameweek_end: int) -> dict[str, pd.DataFrame]:
        """
        Ingests bulk data from a gameweek range
        """
        # warn user if running ingest() without reset
        if self.player_cumulative_stats:
            logger.warning("Cumulative tally has not been reset, proceed with caution.")

        for gw in range(gameweek_start, gameweek_end+1):
            self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)

        return self.player_gw_stats, self.player_cumulative_stats

    def append_gw(self, gw: int) -> dict[str, pd.DataFrame]:
        """
        Ingests a given gameweek
        WARNING: for functionality append_gw() never resets cumulative data,
                 please explicitly call reset() if required.
        """
        self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)
        return self.player_gw_stats, self.player_cumulative_stats