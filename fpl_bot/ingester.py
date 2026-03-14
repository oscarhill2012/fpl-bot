from __future__ import annotations
from .features import Features, FeatureSpec, DataSource
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Callable 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FPLSourceConfig:
    """
    Encodes distinct natures of data sets so all can be treated equally,
    all output columns required must be in map (bar defaults), even if no transform,
    any defaults will be added to map automatically
    """
    provider: DataSource                        # data providers tag, "vaastav", "fci", "opta"
    col_map: dict[str, str]                     # source_col → output columns (should contain all columns even if no change)
    player_id: dict[str, str]                   # maps player_id tag in dataset to "player_id"
    id_map: pd.DataFrame                        # dataframe that contains id inrelation to player code
    stacked: bool                               # one big file vs per-GW files
    denotes_epl: dict[str, str]                 # verifies data from epl match: {indentifying feature: identifying string}
    other_games: bool                           # if dataset contains non-EPL matches, is True
    gw_col: str | None                          # which col has the GW number
    gw_path: str | None                         # path pattern for per-GW files
    transform: dict[str, Callable[[str], any]]  # transform feature


class GameweekProvider:
    """
    Loads per-GW stats from Vaastav/FPL (FPL API) or FPL-Core-Insights (FPL API and Opta).
    Calculates per_90 stats for a given gameweek from cumulative state, which allows calls over multiple gameweeks.
    Reset() used for internal state.
    Outputs a mixture of per_90 and snapshot statistics. 
    """
    def __init__(
        self, 
        features: Features,
        config: FPLSourceConfig,
        season_root: str,
    ):
        self.features = features
        self.cfg = config
        self.season_root = season_root
        self.cumulative_df = None

        # check if stacked df, one initial load saves time
        if self.cfg.stacked:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

        # generate column mapping dicts
        self.cum_map = features.cumulative_map_for(config.provider)
        self.cum_rev = features.inv_cumulative_map_for(config.provider)      

        # generate snapshot, cumulative and output column names for provider
        self.snapshot_cols = features.snapshot_columns_for(self.cfg.provider)
        self.cumulative_cols = features.cumulative_columns_for(self.cfg.provider)
        self.output_cols = list(self.cfg.col_map.values())

        # we add featured column, we want it to be cumulative (but not per_90) and be outputted
        self.output_cols.append("featured")
        self.cumulative_cols.append("featured")

        # columns that will get per_90 rate (subtly different to cumulative_cols as these contain "minutes")
        self.per_90_cols = features.per_90_cols_for(self.cfg.provider)

        # initialise internal state to store points of previous week
        self._prev_points = None
    
    # --- Helper Functions ----
    
    @staticmethod
    def _player_team_index(df: pd.DataFrame) -> pd.Series:
        """
        Composite index: "{player_code}_{team_code}"
        """
        return df["player_code"].astype(int).astype(str) + "_" + df["team_code"].astype(int).astype(str)

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
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Cannot coerce column not in df, {missing} is not")

        # make columns numeric, if can't make 0.0
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return df

    def _add_featured_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds featured = 1, if mins > 0
        """
        df["featured"] = (df["minutes"] > 0).astype(int)
        return df
    
    def _separate_snapshot_cols(self, df: pd.DataFrame, identity_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        
        snapshot_features = df.groupby(identity_col)[self.snapshot_cols].last()
        df = df.drop(columns=self.snapshot_cols)

        return df.reset_index(), snapshot_features

    def _update_cumulative_frame(self, df: pd.DataFrame) -> "GameweekProvider":
        """
        Updates cumulative state dataframe, 
        with df no cumulative tally or by addition if is,
        points are special feature and are directly copied in not added, as recieve cumulative tally.
        Returns:- updated cumulative dataframe state
        """
        # populate cumulative df if none exists, else add to existing df
        # cummulative df has columns given by cum_map
        if self.cumulative_df is None:
            self.cumulative_df = df[self.cumulative_cols].rename(columns=self.cum_map).copy()
        else:
            # extract last total points to difference
            self.cumulative_df = self.cumulative_df.add(df[self.cumulative_cols].rename(columns=self.cum_map), fill_value = 0.0)

        return self

    def _calculate_per_90(self, df: pd.DataFrame, mask: pd.Series, features: list[str]) -> pd.DataFrame:
        """
        Updates per_90 calculation for any feature masked as True (accumulted minutes != 0)
        Intentionally recalulates for players with minutes=0 for a given gw,
        as this is quicker than backfilling empty rows after the fact.
        Returns:- dataframe of same shape as input
        """
        df.loc[mask, features] = (
            self.cumulative_df.loc[mask, self.cum_rev.keys()]
            .rename(columns=self.cum_rev)
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
        Outputs:
                - df with columns of cfg.col_map.values(), plus player_code, 
                  tean_code and indexed by player_team joint index
        
        NOTE: importantly this allows player already in league joining new team to be considered a "new entity"
        """
        # load df
        if self.cfg.stacked:
            # returns a view, so copy
            df = self._stacked_data[self._stacked_data[self.cfg.gw_col] == gw].copy()
        else:    
            df = pd.read_csv(self.season_root + f"GW{gw}/playermatchstats.csv")
        
        # guard: checks all features of col_map are in dataset
        missing = set(self.cfg.col_map.keys()) - set(df.columns)
        if missing:
            raise KeyError(f"GW{gw}: missing source columns: {missing}")

        if self.cfg.other_games is True:
            df = self._match_identifier(df, self.cfg.denotes_epl)

        # if empty, denotes_epl is likely wrong, warn user of this
        if df.empty:
            logger.warning(f"GW{gw}: no data left after filtering for EPL")
        
        # rename to output column names
        df = df.rename(columns=self.cfg.player_id | self.cfg.col_map)

        # guard: coerce all output columns to numeric (real data may have string-typed floats)
        df = self._force_numeric_cols(df, self.output_cols)

        # featured = 1 if player has mins > 0, this will useful for averaging in priors
        df = self._add_featured_col(df)
        
        # aggregate (sums DGW appearances), ignoring ny non-cumulative columns
        df[self.cumulative_cols] = df.groupby("player_id")[self.cumulative_cols].transform("sum")
        df = df.drop_duplicates(subset=["player_id"], keep="last")

        # process df
        df = (
            df  
            .reset_index()                                          # reset index from .groupby      
            .merge(self.cfg.id_map, on="player_id", how="right")    # add all players (including those who didnt play),
            .fillna(0.0)                                            # setting all stats 0 if they didn't feature
            .assign(player_team_id=self._player_team_index)         # adds column "{player_code}_{team_code}"
            .set_index("player_team_id")                            # indexes by new composite index column
            .filter(items=self.output_cols)                         # select only output columns
            .astype(float)                                          # cast all columns as float, player_code safe as index
        )

        # apply transforms
        if self.cfg.transform:
            df = df.assign(**{
                col: func(df[col])
                for col, func in self.cfg.transform.items() if col in df.columns
            })

        self._update_cumulative_frame(df)

        # only per_90ify if minutes not 0, no div by 0
        # uses cumulative df to carry forward players per_90 rates even if they did no feature in gw          
        mask = self.cumulative_df["cum_minutes"] > 0.0
        
        if self.per_90_cols and mask.any():
            df = self._calculate_per_90(df, mask, self.per_90_cols)

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
        features: Features,
        season_root: str,
        fpl_config: FPLSourceConfig,
        opta_config: FPLSourceConfig,
    ):
        self.features = features
        self.season_root = season_root
        self.fpl_provider = GameweekProvider(features, fpl_config, season_root)
        self.opta_provider = GameweekProvider(features, opta_config, season_root)
        self.player_gw_stats: dict[int, pd.DataFrame] = {}
        self.player_cumulative_stats: dict[int, pd.DataFrame] = {}
        self.first_gw: dict[str, int] = {}

        self.cum_rev_map = features.inv_cumulative_map

    # --- Helper Functions ---

    def _combine_dfs(self, fpl: pd.DataFrame, opta: pd.DataFrame) -> pd.DataFrame:
        """
        remove any columns from opta that overlap with fpl, and combine
        NOTE: we prioritise fpl data, when tested minutes differed to 5% in opta but else consistent
        """
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
        gw_combined = self._combine_dfs(
            self.fpl_provider.load_gameweek(gw), 
            self.opta_provider.load_gameweek(gw),
        )
        # extracts cumulative tally attribute from GameweekProvider instance
        gw_cum_combined = self._combine_dfs(
            self.fpl_provider.cumulative_df, 
            self.opta_provider.cumulative_df,
        )
        # ensure that gw_combined has columns ordered by convention
        gw_combined = gw_combined[self.features.pre_sequencer_columns]

        self._update_first_gw(gw_combined, gw)
        return gw_combined, gw_cum_combined

    # --- Public Functions ---

    def reset(self):
        logger.info("FPL and Opta cumulative tallies have been reset.")
        self.fpl_provider.reset()
        self.opta_provider.reset()
        self.player_cumulative_stats = {}
        self.player_gw_stats = {}
        self.first_gw = {}
        return self       

    def ingest(self, gameweek_start: int, gameweek_end: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """ 
        Ingests bulk data from a gameweek range
        """
        # warn user if running ingest() without reset
        if self.player_cumulative_stats:
            logger.warning("Cumulative tally has not been reset, proceed with caution.")

        for gw in range(gameweek_start, gameweek_end+1):
            self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)

        return self.player_gw_stats, self.player_cumulative_stats

    def append_gw(self, gw: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Ingests a given gameweek
        WARNING: for functionality append_gw() never resets cumulative data,
                 please explicitly call reset() if required.
        """
        self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)
        return self.player_gw_stats, self.player_cumulative_stats