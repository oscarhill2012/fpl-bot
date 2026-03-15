from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .features import Features, FeatureSpec, DataSource
from .player_team_index import player_team_index

logger = logging.getLogger(__name__)

@dataclass
class FPLSourceConfig:
    """
    Encodes the distinct nature of a data source so all sources can be treated uniformly.

    All output columns required must be present in col_map (excluding defaults),
    and any defaults will be added to col_map automatically.
    """
    provider: DataSource                        # data providers tag, "vaastav", "fci", "opta"
    col_map: dict[str, str]                     # source_col → output columns (should contain all columns even if no change)
    player_id: dict[str, str]                   # maps player_id tag in dataset to "player_id"
    id_map: pd.DataFrame                        # dataframe that contains id in relation to player code
    stacked: bool                               # one big file vs per-GW files
    denotes_epl: dict[str, str]                 # verifies data from epl match: {identifying feature: identifying string}
    other_games: bool                           # if dataset contains non-EPL matches, is True
    gw_col: str | None                          # which col has the GW number
    gw_path: str | None                         # path pattern for per-GW files
    transform: dict[str, Callable[[str], Any]]  # transform feature


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
    ):
        """
        Initialise a GameweekProvider.

        Args:
            features: Global feature registry.
            config: Source configuration for this provider.
            season_root: Path to the season data directory.
        """
        self.features = features
        self.cfg = config
        self.season_root = season_root
        self.cumulative_gw_data = None

        # check if stacked gw_data, one initial load saves time
        if self.cfg.stacked:
            self._stacked_data = pd.read_csv(self.season_root + self.cfg.gw_path)

        # generate column mapping dicts
        self.cum_map = features.cumulative_map_for(config.provider)
        self.cum_rev_map = features.inv_cumulative_map_for(config.provider)

        # generate snapshot, cumulative and output column names for provider
        # copy lists to avoid mutating the cached properties on the shared Features object
        self.snapshot_cols = features.snapshot_columns_for(self.cfg.provider)
        self.cumulative_cols = list(features.cumulative_columns_for(self.cfg.provider))
        self.output_cols = list(features.pre_sequencer_columns)

        # we add featured column, we want it to be cumulative (but not per_90) and be outputted
        self.output_cols.append("featured")
        self.cumulative_cols.append("featured")

        # columns that will get per_90 rate (subtly different to cumulative_cols as these contain "minutes")
        self.per_90_cols = features.per_90_columns_for(self.cfg.provider)

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
        self.features.validate_dataframe_from(gw_data, self.cfg.provider, f"GW{gw}")

        gw_data = self._match_filter(gw_data, self.cfg.denotes_epl, f"GW{gw}")

        # rename to output column names
        gw_data = gw_data.rename(columns=self.cfg.player_id | self.cfg.col_map)

        # coerce all output columns to numeric (real data may have string-typed floats)
        gw_data = self._force_numeric_cols(gw_data, self.output_cols)

        # featured = 1 if player has mins > 0, this will useful for averaging in priors
        gw_data = self._add_featured_col(gw_data)

        gw_data = self._aggregate_dgw(gw_data)

        # process gw_data
        gw_data = self._join_player_universe(gw_data)

        gw_data = self._apply_transforms(gw_data)

        self._update_cumulative_frame(gw_data)

        # only per_90ify if minutes not 0, no div by 0
        # uses cumulative df to carry forward players per_90 rates even if they did no feature in gw
        mask = self.cumulative_gw_data["cum_minutes"] > 0.0

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
            gw_data = pd.read_csv(self.season_root + f"GW{gw}/playermatchstats.csv")

        return gw_data
    
    def _match_filter(self, gw_data: pd.DataFrame, identifier: dict[str, str], context: str = "") -> pd.DataFrame:
        """If multiple matches, filter DataFrame rows for key column contains value string."""
        if self.cfg.other_games:
            for col, string in identifier.items():
                # regex ensures string can contain special characters
                gw_data = gw_data[gw_data[col].str.contains(string, regex=False)]

        # if empty, filter is likely wrong, warn user of this
        if gw_data.empty:
            logger.warning(f"{context}: no data left after filtering for EPL")

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
        gw_data["featured"] = (gw_data["minutes"] > 0).astype(int)

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
            .assign(player_team_index)                              # adds column "{player_code}_{team_code}"
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
    ):
        """
        Initialise the Ingester with FPL and Opta data providers.

        Args:
            features: Global feature registry.
            season_root: Path to the season data directory.
            fpl_config: Source configuration for the FPL/Vaastav data.
            opta_config: Source configuration for the Opta data.
        """
        self.features = features
        self.season_root = season_root
        self.fpl_provider = GameweekProvider(features, fpl_config, season_root)
        self.opta_provider = GameweekProvider(features, opta_config, season_root)
        self.player_gw_stats: dict[int, pd.DataFrame] = {}
        self.player_cumulative_stats: dict[int, pd.DataFrame] = {}
        self.first_gw: dict[str, int] = {}

        self.cum_rev_map = features.inv_cumulative_map

    #================================================
    # Public Functions
    #================================================

    def reset(self) -> "Ingester":
        """Reset all cumulative tallies and cached data; use between season reloads."""
        logger.info("FPL and Opta cumulative tallies have been reset.")
        self.fpl_provider.reset()
        self.opta_provider.reset()
        self.player_cumulative_stats = {}
        self.player_gw_stats = {}
        self.first_gw = {}
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
    # Private Helpers
    #================================================

    def _combine_dfs(self, fpl: pd.DataFrame, opta: pd.DataFrame) -> pd.DataFrame:
        """Merge FPL and Opta DataFrames, dropping overlapping Opta columns to give FPL priority."""
        overlap = fpl.columns.intersection(opta.columns)
        opta = opta.drop(columns=overlap, errors="ignore")

        return pd.concat([fpl, opta], axis=1)

    def _update_first_gw(self, gw_combined: pd.DataFrame, gw: int) -> "Ingester":
        """Record the first gameweek each player appeared in."""
        # isolate players that played this gameweek
        played = gw_combined.index[gw_combined["minutes"] > 0]

        # pandas search in C so quicker than if statement
        new_players = played.difference(self.first_gw.keys())

        self.first_gw.update({pc: gw for pc in new_players})
        return self

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
        gw_combined = gw_combined[self.features.pre_sequencer_columns]

        self._update_first_gw(gw_combined, gw)
        return gw_combined, gw_cum_combined
