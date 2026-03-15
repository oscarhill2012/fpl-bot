from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
from torch._C import int16
from torch.utils.data import Dataset
from .ingester import Ingester, FPLSourceConfig
from .priors import PriorComputer, PriorData
from .features import Features, FeatureSpec, FeatureType
from .player_team_index import player_team_index


import logging
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
    which ingests one new GW and updates all internal derived state (carry-forward, first appearances).

    Typical lifecycle:
        seq = SeasonSequencer(...)
        seq.ingest_range(1, 30)       # bulk load GW1-30
        ds = seq.dataset(1, 30)       # torch Dataset for training
        seq.step()                    # ingest GW31
        preds = seq.predict_batch(32) # predict GW32
    """
    def __init__(
        self,
        features: Features,
        season_root: str,
        fpl_config_season: FPLSourceConfig,
        opta_config: FPLSourceConfig,
        player_meta: pd.DataFrame,
        fixture_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        window_size: int=8,
        prior_data: PriorData | None = None,
    ):
        """
        Initialise the SeasonSequencer.

        Args:
            features: Global feature registry.
            season_root: Path to the season data directory.
            fpl_config_season: Source configuration for the FPL data provider.
            opta_config: Source configuration for the Opta data provider.
            player_meta: DataFrame indexed by player_team_code with columns
                including 'position' and 'team_code'.
            fixture_df: DataFrame of fixtures for the season.
            teams_df: DataFrame of team strength and ELO data.
            window_size: Number of past gameweeks to include in each sequence.
            prior_data: Pre-computed prior data; if None, computed from the
                first ingest_range() call.
        """
        self.season_root = season_root
        self._ingester = Ingester(features, season_root, fpl_config_season, opta_config)
        self._window_size = window_size

        # generate list of categorical and temporal features
        self.features = features
        self._temporal_features = features.temporal_columns
        self._catergorical_features = features.categorical_columns

        # index by player code for fast lookup
        if player_meta.index.name != "player_team_id":
            meta = player_meta.copy()
            meta["player_team_id"] = player_team_index(meta)   # shared util
            self.player_meta = meta.set_index("player_team_id")
        else:
            meta = player_meta.copy()

        # save player_meta as a dict for quicker lookup
        self._player_meta: dict[str, dict[str, int]] = meta.to_dict(orient="index")


        # nothing ingested yet
        self._current_gameweek: int = 0
        self._prior_data = prior_data

        # first gameweek a player plays minutes > 0, until then we use priors
        self._first_gw: dict[str, str] = {}

        # find fixtures, (team_code, gw) -> fixture info
        self._fixture_lookup: dict[tuple[int, str], dict] = {}

        # cache data in as dict, so lookup cheaper than pandas .loc
        self.gw_cache: dict[int, dict[str, dict[str, float]]] = {}

    #================================================
    # Ingestion and State Management
    # Methods for loading gameweeks and advancing state.
    #================================================

    def ingest_range(self, gw_start: int, gw_end: int) -> "SeasonSequencer":
        """
        Bulk-ingest a range of gameweeks.

        Computes prior data from the ingested range if none has been provided.

        Args:
            gw_start: First gameweek to ingest (inclusive).
            gw_end: Last gameweek to ingest (inclusive).

        Returns:
            Self, for method chaining.
        """
        # reset ingester to ensure clean cumulative states
        self._ingester.reset()

        self._ingester.ingest(gw_start, gw_end)
        self._first_gw = self._ingester.first_gw
        self._current_gameweek = gw_end

        # add new data to gw_cache
        for gw in range(gw_start, gw_end+1):
            self._cache_gw(gw)

        if self._prior_data is None:
            self._prior_data = PriorData.from_data(
                self.features,
                self._ingester.player_gw_stats,
                self._ingester.player_cumulative_stats,
                pd.DataFrame.from_dict(self._player_meta, orient="index"),
                min_mins=450.0
            )
            self._prior_data.to_json(self.season_root)

        return self

    def step(self, teams_df: pd.DataFrame | None = None) -> "SeasonSequencer":
        """
        Advance the sequencer by one gameweek.

        Uses the ingester's append_gw(). Optionally refreshes team strength and ELO.

        Args:
            teams_df: Optional updated team strength DataFrame.

        Returns:
            Self, for method chaining.

        Raises:
            RuntimeError: If called before priors have been computed or provided.
        """
        # require priors to step
        if self._prior_data is None:
            raise RuntimeError("Please provide priors or ingest_range() to step()")

        self._current_gameweek += 1

        self._ingester.append_gw(self._current_gameweek)
        self._first_gw = self._ingester.first_gw

        self._cache_gw(self._current_gameweek)

        if teams_df is not None:
            self._update_team_strength(teams_df)

        return self

    #================================================
    # Sequence Construction
    # Methods that build fixed-length player sequences
    # from cached gameweek data and prior estimates.
    #================================================

    def build_player_window(self, player_team_id: str, window: list[int], inference: bool = False) -> dict:
        """
        Construct a fixed-length sequence of gameweek rows for a single player.

        For each timestep, selects either real data or a prior row depending on
        whether the player had appeared in the data by that gameweek.

        Args:
            player_team_id: Composite player-team identifier, e.g. "12345_99".
            window: List of gameweek numbers to include in the sequence.
            inference: Whether the sequence is being built for inference.

        Returns:
            Dict containing the assembled sequence data.
        """
        position = self._player_meta[player_team_id]["position"]
        team_code = self._player_meta[player_team_id]["team_code"]
        first_gw = self._first_gw[player_team_id]
        gw_window: list[int] = []

        # filter any gameweeks before the player's first appearance
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
                prior_row["data_age"] = self._window_size - i
                data_list.append(list(prior_row.values()))
            else:
                real_row = self._get_real_row(player_team_id, gw)
                data_list.append(list(real_row.values()))
                self._verify_consistent_features(list(real_row.keys()), self._temporal_features + ["is_prior", "data_age"])

        # convert to 2D numpy array, shape [n_timesteps, n_features]
        x_temporal = np.array(data_list, dtype=np.float32)

    #================================================
    # Private Helpers
    #================================================

    def _cache_gw(self, gw: int) -> "SeasonSequencer":
        """Convert the ingested DataFrame for a gameweek to a nested dict for fast lookup."""
        self.gw_cache[gw] = self._ingester.player_gw_stats[gw].to_dict(orient="index")
        return self

    def _update_team_strength(self, teams_df: pd.DataFrame) -> "SeasonSequencer":
        """Update team strength and ELO from a new teams DataFrame."""
        self._team_strength = self._parse_team_strength(teams_df)
        logger.info("Updated team strength")

        return self

    def _build_window(self, target_gw: int) -> list[int]:
        """Build the list of gameweeks in the sliding window ending before target_gw."""
        # window must always start at seq_start > 0
        seq_start = max(1, target_gw - self._window_size)

        return list(range(seq_start, target_gw))

    def _get_prior(self, player_team_id: str, team_code: str, position: str) -> dict[str, float]:
        """Look up the best available prior for a player, falling back through the hierarchy."""
        if self._prior_data is None:
            raise RuntimeError("No prior data available, call ingeste_range() first.")

        pos_team = "_".join([position, team_code])       # NOTE: order is convention

        # lookup player in prior hierarchy
        # load dict as shallow copy, so we dont mutate
        if player_team_id in self._prior_data.individual:
            prior_row = {**self._prior_data.individual[player_team_id]}

        elif pos_team in self._prior_data.position_team:
            prior_row = {**self._prior_data.position_team[pos_team]}
            prior_row["minutes"] = self._minutes_lookup(prior_row)

        elif position in self._prior_data.position:
            prior_row = {**self._prior_data.position[position]}
            prior_row["minutes"] = self._minutes_lookup(prior_row)

        else:
            prior_row = {**self._prior_data.league["league"]}
            prior_row["minutes"] = self._minutes_lookup(prior_row)

        # add row to denote prior, "data_age" is sentinel
        prior_row["is_prior"], prior_row["data_age"] = 1.0, 0.0

        return prior_row

    @staticmethod
    def _minutes_lookup(prior_row: dict[str, float]) -> float:
        """Return a scaled minutes estimate for a player with no recorded data; currently a stub returning 0.0."""
        return 0.0

    def _get_real_row(self, player_team_code: str, gw: int) -> dict[str, float]:
        """Retrieve a player's cached stats for a given gameweek."""
        # player lookup in cache
        real_row = self.gw_cache[gw][player_team_code]

        # add row to denote prior, "data_age" is sentinel
        real_row["is_prior"], real_row["data_age"] = 0.0, 0.0
        return real_row

    def _verify_consistent_features(self, list1: list[str], list2: list[str]) -> "SeasonSequencer":
        """Raise RuntimeError if two feature name lists differ."""
        if list1 != list2:
            raise RuntimeError("Inconsistent features between prior and current data")
        return self
