from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import numpy as np 

from .features import Features, FeatureSpec, DataSource
from .player_team_index import player_team_index

logger = logging.getLogger(__name__)


def _match_filter(
    gw_data: pd.DataFrame,
    other_games: bool,
    identifier: dict[str, str],
    context: str = "",
) -> pd.DataFrame:
    """Filter DataFrame rows where the identifying column contains the expected string."""
    if other_games:
        for col, string in identifier.items():
            # regex=False ensures string can contain special characters
            gw_data = gw_data[gw_data[col].str.contains(string, regex=False)]

    if gw_data.empty:
        logger.warning(f"{context}: no data left after filtering for EPL")

    return gw_data


@dataclass
class FPLSourceConfig:
    """
    Encodes the distinct nature of a data source so all sources can be treated uniformly.
    """
    provider: DataSource                                # data providers tag, "vaastav", "fci", "opta", "fixture"
    player_id: dict[str, str]                           # maps player_id column name in dataset to "player_id"
    id_map: pd.DataFrame                                # dataframe that contains local identity to global identity map
    stacked: bool                                       # one big file vs per-GW files
    denotes_epl: dict[str, str]                         # verifies data from epl match: {identifying feature: identifying string}
    other_games: bool                                   # if dataset contains non-EPL matches, is True
    gw_col: str | None                                  # which col has the GW number
    gw_path: str | None                                 # path pattern for per-GW files
    gw_filename: str | None                              # per-GW CSV filename (e.g. "playerstats.csv")
    transform: dict[str, Callable[[str], Any]] | None   # transform feature

@dataclass
class FixtureSourceConfig:
    """
    Encodes the distinct nature of a fixture source so all sources can be treated uniformly.
    """
    provider: DataSource                                # data providers tag, "vaastav", "fci", "opta", "fixture"
    team_codes: pd.DataFrame
    stacked: bool                                       # one big file vs per-GW files
    denotes_epl: dict[str, str]                         # verifies data from epl match: {identifying feature: identifying string}
    other_games: bool                                   # if dataset contains non-EPL matches, is True
    gw_col: str | None                                  # which col has the GW number
    gw_path: str | None                                 # path pattern for per-GW files
    gw_filename: str                                    # per-GW CSV filename (e.g. "matches.csv")


class FixtureProvider:
    def __init__(
        self,
        features: Features,
        config: FixtureSourceConfig,
        season_root: str,
    ):
        """
        Initialise a FixtureProvider.

        Args:
            features: Global feature registry.
            config: Source configuration for this provider.
            season_root: Path to the season data directory.
        """
        self.features = features
        self.cfg = config
        self.season_root = season_root
        self.providers = [self.cfg.provider, DataSource.FIXINGESTER]

        # output_cols: features (ordered by Feature convention) output from provider instance
        self.output_cols = features.output_columns_for(self.providers)

        # provider_cols: features this provider loads from CSV (used for numeric coercion)
        self._provider_cols = list(features.output_columns_for(self.cfg.provider))

        # derive_cols: any features to be derived in provider
        self._derived_cols = list(features.output_columns_for([DataSource.FIXINGESTER]))

    #================================================
    # Public Functions
    #================================================

    def load_fixtures(self, gw: int) -> pd.DataFrame:
        """
        Loads fixture data for one gameweek, from given FixtureSource.

        Args:
            gw: which gameweek to load
        
        Returns:
            DataFrame with fixture_columns from features
        """
        # rename columns so they end at team_code when processing
        raw_fixtures = (
            pd.read_csv(self.season_root + self.cfg.gw_path + f"GW{gw}/{self.cfg.gw_filename}")
            .rename(columns={"home_team": "home_team_code", "away_team": "away_team_code"})
        )
        raw_fixtures = _match_filter(raw_fixtures, self.cfg.other_games, self.cfg.denotes_epl, f"GW{gw}")
        
        # dataset gives one row per game (teams are home and away)
        # we want one row per team, so we need to split and rename columns
        home_df, away_df = self._home_away_split(raw_fixtures)
        home_df, away_df = self._remap_col_names(home_df, away_df)

        # is_home introduced to explicitly convey if team home or away
        home_df, away_df = self._add_is_home(home_df, away_df)

        # stack home and away teams, giving one row per team that played
        gw_fixtures = pd.concat([home_df, away_df], axis=0).copy()

        if self._derived_cols:
            gw_fixtures = self._add_derived_cols(gw_fixtures)

        # if team didnt play add in blank row, 
        # is_home = -1 (0 = away, 0 != not playing)
        gw_fixtures = self._join_team_universe(gw_fixtures)

        gw_fixtures = (
            gw_fixtures
            .filter(items=self.output_cols)         # select only output columns
            .round()                                # round any ELOs
            .astype(np.int64)                       # set as ints       
        )

        gw_fixtures = self._flatten_dgw(gw_fixtures)

        self.features.validate_dataframe_from_source(
            gw_fixtures, self.output_cols, [self.cfg.provider], context=f"GW{gw}"
        )
        gw_fixtures = gw_fixtures[self.output_cols]

        return gw_fixtures

    #================================================   
    # Private Helpers
    #================================================   

    @staticmethod
    def _home_away_split(gw_fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate home and away features into two DataFrames."""
        home_columns = [col for col in gw_fixtures.columns if "home" in col]
        away_columns = [col for col in gw_fixtures.columns if "away" in col]

        home_df = gw_fixtures[home_columns]
        away_df = gw_fixtures[away_columns]
        
        return home_df, away_df 

    @staticmethod
    def _remap_col_names(
        home_df: pd.DataFrame,
        away_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strip home_/away_ prefixes and add oppo_ prefix for opponent columns."""
        # strip prefixes to get neutral column names
        home_neutral = {col: col.replace("home_", "") for col in home_df.columns}
        away_neutral = {col: col.replace("away_", "") for col in away_df.columns}

        # for home team: own stats are home columns, opponent stats are away columns
        home_own = home_df.rename(columns=home_neutral)
        home_oppo = away_df.rename(
            columns={col: f"oppo_{neutral}" for col, neutral in away_neutral.items()},
        )
        home_combined = pd.concat([home_own, home_oppo], axis=1)

        # for away team: own stats are away columns, opponent stats are home columns
        away_own = away_df.rename(columns=away_neutral)
        away_oppo = home_df.rename(
            columns={col: f"oppo_{neutral}" for col, neutral in home_neutral.items()},
        )
        away_combined = pd.concat([away_own, away_oppo], axis=1)

        return home_combined.copy(), away_combined.copy()
    
    @staticmethod
    def _add_is_home(home_df: pd.DataFrame, away_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add is_home column: 1 for home team, 0 for away team."""
        home_df["is_home"] = 1
        away_df["is_home"] = 0

        return home_df, away_df

    def _add_derived_cols(self, gw_fixtures: pd.DataFrame) -> pd.DataFrame:
        """Requests specific derivations for fixture-ingester features."""
        for feature in self._derived_cols:
            if feature == "num_matches":
                gw_fixtures = self._derive_num_matches(gw_fixtures, feature)

        return gw_fixtures
    
    def _join_team_universe(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Right-join on team_codes so every known team has a row. -1 for missing is_home."""
        return (
            fixtures
            .merge(self.cfg.team_codes, on="team_code", how="right")  # ensure every team has a row
            .fillna({"is_home": -1})                                  # -1 signals team didn't play this GW
            .fillna(0)
        )

    def _flatten_dgw(self, gw_fixtures: pd.DataFrame) -> pd.DataFrame:
        """Group by team_code and average for DGWs; sum-aggregated columns use sum instead."""
        # columns that should be summed rather than averaged (e.g. num_matches: 1+1 = 2 in a DGW)
        sum_cols = [c for c in self._derived_cols if c in gw_fixtures.columns]
        avg_cols = [c for c in gw_fixtures.columns if c not in sum_cols and c != "team_code"]

        grouped = gw_fixtures.groupby("team_code", as_index=False)
        agg_map = {c: "mean" for c in avg_cols}
        agg_map.update({c: "sum" for c in sum_cols})

        gw_fixtures = (
            grouped.agg(agg_map)
            .set_index("team_code", drop=False)
        )
        return gw_fixtures

    #================================================
    # Feature Derivation Methods
    #================================================

    @staticmethod
    def _derive_num_matches(gw_fixtures: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Stamp each fixture row with 1; DGW flatten sums to 2 for double gameweeks."""
        gw_fixtures[feature] = 1
        return gw_fixtures

class GameweekProvider:
    """
    Loads per-GW stats from Vaastav/FPL (FPL API) or FPL-Core-Insights (FPL API and Opta).

    Calculates per_90 stats for a given gameweek from cumulative state, which allows calls
    over multiple gameweeks. reset() is used to clear internal state between seasons.
    Outputs a mixture of per_90 and snapshot statistics.
    """
    def __init__(
        self,
        features: Features,
        config: FPLSourceConfig,
        season_root: str,
        minimum_minutes: int = 45,
    ):
        """
        Initialise a GameweekProvider.

        Args:
            features: Global feature registry.
            config: Source configuration for this provider.
            season_root: Path to the season data directory.
            minimum_minutes: Cumulative minutes threshold before per-90 rates
                are calculated. Players below this threshold keep per-90
                columns at zero to avoid extreme outliers.
        """
        self.features = features
        self.cfg = config
        self.season_root = season_root
        self.minimum_minutes = minimum_minutes
        self.cumulative_gw_data = None
  
        # providers: all the features.source contained in this class
        # derive_cols: any features to be derived in provider
        if self.cfg.provider == DataSource.OPTA:
            self.providers = [self.cfg.provider, DataSource.OPTAINGESTER]  
            self._derived_cols = list(features.output_columns_for([DataSource.OPTAINGESTER])) 
        else:
            self.providers = [self.cfg.provider, DataSource.FPLINGESTER] 
            self._derived_cols = list(features.output_columns_for([DataSource.FPLINGESTER])) 
        
        # check if stacked gw_data, one initial load saves time
        if self.cfg.stacked:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

        # generate column mapping dicts
        self.cum_map = features.cumulative_map_for(self.providers)
        self.cum_rev_map = features.inv_cumulative_map_for(self.providers)
        self.source_map = features.source_map(self.cfg.provider)


        # generate snapshot, cumulative and output column names for provider
        # copy lists to avoid mutating the cached properties on the shared Features object
        self.snapshot_cols = features.snapshot_columns_for(self.providers)
        self.cumulative_cols = list(features.cumulative_columns_for(self.providers))

        # output_cols: features (ordered by Feature convention) output from provider instance
        self.output_cols = features.output_columns_for(self.providers)

        # provider_cols: features this provider loads from CSV (used for numeric coercion)
        self._provider_cols = list(features.output_columns_for(self.cfg.provider))

        # NOTE: featured is a synthetic internal only feature, so manually add
        self.output_cols.append("featured")
        self.cumulative_cols.append("featured")
        self.cum_map["featured"] = "cum_featured"
        self.cum_rev_map["cum_featured"] = "featured"

        # columns that will get per_90 rate (subtly different to cumulative_cols as these contain "minutes")
        self.per_90_cols = features.per_90_columns_for(self.providers)

    #================================================
    # Public Functions
    #================================================

    def reset(self) -> "GameweekProvider":
        """Reset internal cumulative state; must be called between seasons."""
        self.cumulative_gw_data = None
        return self

    def load_gameweek(
        self,
        gw: int,
        ) -> pd.DataFrame:
        """
        Load, process, and return stats for a single gameweek.

        Calculates per_90 features from the running cumulative state, and stores
        the updated cumulative tally internally.

        Args:
            gw: Gameweek number to load.

        Returns:
            DataFrame with columns from cfg.col_map.values(), plus player_code,
            team_code, indexed by the composite player_team_id index.

        Note:
            This treats a player who joins a new team mid-season as a new entity.
        """
        gw_data = self._load_raw(gw)

        # validate dataframe
        self.features.validate_dataframe_from_source(gw_data, self._provider_cols, [self.cfg.provider],f"GW{gw}")

        gw_data = _match_filter(gw_data, self.cfg.other_games, self.cfg.denotes_epl, f"GW{gw}")

        gw_data = self._per_90_guard(gw_data)

        # add derived columns, if any
        if self._derived_cols:
            gw_data = self._add_derived_columns(gw_data)

        # rename to output column names
        gw_data = gw_data.rename(columns=self.cfg.player_id | self.source_map)
   
        # coerce all output columns to numeric (real data may have string-typed floats)
        gw_data = self._force_numeric_cols(gw_data, self._provider_cols)

        # featured = 1 if player has mins > 0, this will useful for averaging in priors
        gw_data = self._add_featured_col(gw_data)
        
        gw_data = self._aggregate_dgw(gw_data)

        # process gw_data
        gw_data = self._join_player_universe(gw_data)

        gw_data = self._apply_transforms(gw_data)

        self._update_cumulative_frame(gw_data)

        # only per_90ify if player has accrued enough cumulative minutes
        # suppresses extreme outliers from low-minute players
        mask = self.cumulative_gw_data["cum_minutes"] >= self.minimum_minutes

        if self.per_90_cols and mask.any():
            gw_data = self._calculate_per_90(gw_data, mask, self.per_90_cols)

        return gw_data

    #================================================
    # Private Helpers
    #================================================

    def _load_raw(self, gw: int) -> pd.DataFrame:
        """Load raw CSV for one GW, validate source columns exist."""
        if self.cfg.stacked:
            # returns a view, so copy
            gw_data = self._stacked_data[self._stacked_data[self.cfg.gw_col] == gw].copy()
        else:
            prefix = self.cfg.gw_path or ""
            gw_data = pd.read_csv(self.season_root + prefix + f"GW{gw}/{self.cfg.gw_filename}")
            gw_data.attrs["source"] = self.cfg.gw_filename

        return gw_data
    
    def _per_90_guard(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove any existing per_90 columns so no duplicates arise when we rename."""
        for col in gw_data.columns:
            if "per_90" in col:
                gw_data = gw_data.drop(col, axis=1)

        return gw_data

    def _force_numeric_cols(self, gw_data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Coerce specified columns to float, filling non-numeric values with 0.0."""
        missing = set(cols) - set(gw_data.columns)

        if missing:
            raise ValueError(f"Columns {missing} not found in gameweek data.")
        
        # make columns numeric, if can't make 0.0
        gw_data[cols] = gw_data[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return gw_data

    def _add_featured_col(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Add a binary 'featured' column; 1 if the player had minutes > 0."""
        gw_data["featured"] = (gw_data["minutes"] > 0).astype(np.int64)

        return gw_data

    def _add_derived_columns(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Checks which features need deriving, requests derivation."""
        # please see: Feature Derivation Methods
        for feature in self._derived_cols:
            if feature == "defcon_per_90":
                gw_data = self._defcon_derivation(gw_data, feature)

        return gw_data

    def _aggregate_dgw(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate (sums DGW appearances), ignoring any non-cumulative columns"""
        gw_data[self.cumulative_cols] = gw_data.groupby("player_id")[self.cumulative_cols].transform("sum")
        gw_data = gw_data.drop_duplicates(subset=["player_id"], keep="last")

        return gw_data

    def _join_player_universe(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Right-join on id_map so every known player has a row. 0.0 for missing data."""
        return (
            gw_data
            .reset_index()                                          # reset index from .groupby
            .merge(self.cfg.id_map, on="player_id", how="right")    # add all players (including those who didnt play),
            .fillna(0.0)                                            # setting all stats 0 if they didn't feature
            .assign(player_team_id=player_team_index)               # adds column "{player_code}_{team_code}"
            .set_index("player_team_id")                            # indexes by new composite index column
            .filter(items=self.output_cols)                         # select only output columns
            .astype(float)                                          # cast all columns as float, player_code safe as index
        )

    def _apply_transforms(self, gw_data: pd.DataFrame) -> pd.DataFrame:
        """Apply any column transforms in config."""
        if self.cfg.transform:
            gw_data = gw_data.assign(**{
                col: func(gw_data[col])
                for col, func in self.cfg.transform.items() if col in gw_data.columns
            })
        return gw_data

    def _update_cumulative_frame(self, gw_data: pd.DataFrame) -> "GameweekProvider":
        """Add the current gameweek's stats to the running cumulative tally."""
        # populate cumulative gw_data if none exists, else add to existing gw_data
        # cumulative gw_data has columns given by cum_map
        if self.cumulative_gw_data is None:
            self.cumulative_gw_data = gw_data[self.cumulative_cols].rename(columns=self.cum_map).copy()
        else:
            # extract last total points to difference
            self.cumulative_gw_data = self.cumulative_gw_data.add(
                gw_data[self.cumulative_cols].rename(columns=self.cum_map),
                fill_value=0.0,
            )

        return self

    def _calculate_per_90(self, gw_data: pd.DataFrame, mask: pd.Series, features: list[str]) -> pd.DataFrame:
        """Update per-90 rates for rows with non-zero cumulative minutes."""
        gw_data.loc[mask, features] = (
            self.cumulative_gw_data.loc[mask, self.cum_rev_map.keys()]
            .rename(columns=self.cum_rev_map)
            .div(self.cumulative_gw_data.loc[mask, "cum_minutes"] / 90, axis=0)
        )
        return gw_data

    #================================================
    # Feature Derivation Methods
    #================================================

    @staticmethod
    def _defcon_derivation(gw_data: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Sum tackles, recoveries, blocks, and clearances into the defcon feature."""
        gw_data[feature] = gw_data[["tackles", "recoveries", "blocks", "clearances"]].sum(axis=1)

        return gw_data
    
class Ingester:
    """
    Ingests data from a source depending on the provided config.

    Outputs both snapshot and cumulative data; output stored in dict.

    Convention:
        - Only to be used for one season at a time.
        - Output dict is {int: DataFrame} keyed by gameweek number.

    Warning:
        Both Opta and FPL API contain minutes that slightly differ.
        Opta minutes are used for Opta per-90 calculations; when tested,
        differences were at most 5% over a season.
    """
    def __init__(
        self,
        features: Features,
        season_root: str,
        fpl_config: FPLSourceConfig,
        opta_config: FPLSourceConfig,
        fixture_config: FixtureSourceConfig,
        minimum_minutes: int = 45,
    ):
        """
        Initialise the Ingester with FPL and Opta data providers.

        Args:
            features: Global feature registry.
            season_root: Path to the season data directory.
            fpl_config: Source configuration for the FPL/Vaastav data.
            opta_config: Source configuration for the Opta data.
            fixture_config: Source configuration for fixture/Elo data.
            minimum_minutes: Cumulative minutes threshold before per-90 rates
                are calculated. Passed to each GameweekProvider.
        """
        self.features = features
        self.season_root = season_root
        
        self.player_providers = [
            fpl_config.provider, 
            opta_config.provider,  
            DataSource.OPTAINGESTER, 
            DataSource.FPLINGESTER,
        ]
        self.fixture_providers = [
            fixture_config.provider,
            DataSource.FIXINGESTER,
        ]

        self.player_output_cols = features.output_columns_for(self.player_providers)
        self.fixture_output_cols = features.output_columns_for(self.fixture_providers)
        # initialise providers
        self.fpl_provider = GameweekProvider(features, fpl_config, season_root, minimum_minutes)
        self.opta_provider = GameweekProvider(features, opta_config, season_root, minimum_minutes)
        self.fixture_provider = FixtureProvider(features, fixture_config, season_root)

        # assign states for data storage
        self.player_gw_stats: dict[int, pd.DataFrame] = {}
        self.player_cumulative_stats: dict[int, pd.DataFrame] = {}
        self.fixtures: dict[int, pd.DataFrame] = {}

        # players given first week 39 if they haven't played yet 
        player_team_codes = fpl_config.id_map.copy()
        player_team_codes = player_team_codes.assign(player_team_id=player_team_index)
        player_team_codes["first_gw"] = 39
        self._init_first_gw: dict[str, int] = dict(
            zip(player_team_codes['player_team_id'], player_team_codes['first_gw'])
        )
        self.first_gw = self._init_first_gw.copy()

    #================================================
    # Player Data Ingestion
    #================================================

    def reset(self) -> "Ingester":
        """Reset all cumulative tallies and cached data; use between season reloads."""
        logger.info("FPL and Opta cumulative tallies have been reset.")
        self.fpl_provider.reset()
        self.opta_provider.reset()
        self.player_cumulative_stats = {}
        self.player_gw_stats = {}

        self.first_gw = self._init_first_gw.copy()
        return self

    def ingest(self, gameweek_start: int, gameweek_end: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Ingest bulk data for a range of gameweeks.

        Args:
            gameweek_start: First gameweek to ingest (inclusive).
            gameweek_end: Last gameweek to ingest (inclusive).

        Returns:
            Tuple of (player_gw_stats, player_cumulative_stats) dicts.
        """
        # warn user if running ingest() without reset
        if self.player_cumulative_stats:
            logger.warning("Cumulative tally has not been reset, proceed with caution.")

        for gw in range(gameweek_start, gameweek_end+1):
            self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)

        return self.player_gw_stats, self.player_cumulative_stats

    def append_gw(self, gw: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """
        Ingest a single gameweek, appending to existing state.

        Warning:
            append_gw() never resets cumulative data.
            Call reset() explicitly if a clean state is required.

        Args:
            gw: Gameweek number to ingest.

        Returns:
            Tuple of (player_gw_stats, player_cumulative_stats) dicts.
        """

        self.player_gw_stats[gw], self.player_cumulative_stats[gw] = self._process_gw(gw)
        return self.player_gw_stats, self.player_cumulative_stats

    #================================================
    # Feature Ingestion
    #================================================

    def ingest_fixtures_range(self, gameweek_start: int, gameweek_end: int) -> dict[int, pd.DataFrame]:
        """
        Ingest fixture data for a range of gameweeks.

        Args:
            gameweek_start: First gameweek to ingest (inclusive).
            gameweek_end: Last gameweek to ingest (inclusive).

        Returns:
            Dict of {gameweek: fixture DataFrame}.
        """
        for gw in range(gameweek_start, gameweek_end + 1):
            self.fixtures[gw] = self.fixture_provider.load_fixtures(gw)[self.fixture_output_cols]

        return self.fixtures

    def update_future_fixtures(self, current_gameweek: int, final_gameweek: int = 38) -> "Ingester":
        """
        Re-ingest future fixtures, from current gameweek until end of season.

        Distinct from ingest_fixtures_range because Elo ratings in fixture files
        change after results come in. Only future gameweeks should be refreshed;
        updating past fixtures would leak result information into the model.

        Args:
            current_gameweek: First gameweek to refresh (inclusive).
            final_gameweek: Last gameweek to refresh (inclusive), defaults to 38.

        Returns:
            Self, to make the state mutation explicit.
        """
        for gw in range(current_gameweek, final_gameweek + 1):
            self.fixtures[gw] = self.fixture_provider.load_fixtures(gw)

        return self

    #================================================
    # Private Helpers
    #================================================

    def _combine_dfs(self, fpl: pd.DataFrame, opta: pd.DataFrame) -> pd.DataFrame:
        """Merge FPL and Opta DataFrames, dropping overlapping Opta columns to give FPL priority."""
        overlap = fpl.columns.intersection(opta.columns)
        opta = opta.drop(columns=overlap, errors="ignore")

        return pd.concat([fpl, opta], axis=1)

    def _process_gw(self, gw: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and merge FPL and Opta data for a single gameweek."""
        # uses instances of GameweekProvider to load gw
        gw_combined = self._combine_dfs(
            self.fpl_provider.load_gameweek(gw),
            self.opta_provider.load_gameweek(gw),
        )
        # extracts cumulative tally attribute from GameweekProvider instance
        gw_cum_combined = self._combine_dfs(
            self.fpl_provider.cumulative_gw_data,
            self.opta_provider.cumulative_gw_data,
        )
        # ensure that gw_combined has columns ordered by convention
        gw_combined = gw_combined[self.player_output_cols]

        self._update_first_gw(gw_combined, gw)
        return gw_combined, gw_cum_combined

    def _update_first_gw(self, gw_combined: pd.DataFrame, gw: int) -> "Ingester":
        """Record the first gameweek each player appeared in."""
        # isolate players that played this gameweek
        played_this_gw = gw_combined.index[gw_combined["minutes"] > 0]

        # players that haven't played are labelled with first gw = 39
        played_previously = {key: value for key, value in self.first_gw.items() if value != 39}
  
        # pandas search in C so quicker than if statement
        new_players = played_this_gw.difference(played_previously.keys())

        for pc in new_players:
            self.first_gw[pc] = gw

        return self
