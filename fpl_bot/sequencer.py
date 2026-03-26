from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import Features, FeatureSpec, FeatureType, DataSource
from .ingester import Ingester, FPLSourceConfig, FixtureSourceConfig
from .player_team_index import player_team_index
from .priors import PriorData

logger = logging.getLogger(__name__)


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
        fixture_config: FixtureSourceConfig,
        player_meta: pd.DataFrame,
        window_size: int=8,
        predict_window_size: int=1,
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
            fixture_config: Source configuration for fixture data.
            window_size: Number of past gameweeks to include in each sequence.
            prior_data: Pre-computed prior data; if None, computed from the
                first ingest_range() call.
        """
        self.season_root = season_root
        self._ingester = Ingester(features, season_root, fpl_config_season, opta_config, fixture_config)
        self._window_size = window_size
        self._target_window_size = predict_window_size
        self.features = features

        self.providers = [
            DataSource.OPTA, 
            DataSource.FCI, 
            DataSource.VAASTAV, 
            DataSource.OPTAINGESTER, 
            DataSource.FPLINGESTER,
            DataSource.PRIOR, 
            DataSource.SEQUENCER
        ]

        self._output_columns = features.output_columns_for(self.providers)
        self._derived_columns = features.output_columns_for([DataSource.SEQUENCER])
        
        # index by player code for fast lookup
        if player_meta.index.name != "player_team_id":
            meta = player_meta.copy()
            meta["player_team_id"] = player_team_index(meta)   # shared util
            self.player_meta = meta.set_index("player_team_id")
        else:
            meta = player_meta.copy()
            self.player_meta = meta

        # save player_meta as a dict for quicker lookup
        self._player_meta: dict[str, dict[str, int]] = self.player_meta.to_dict(orient="index")

        # nothing ingested yet
        self._current_gameweek: int = 0
        self._prior_data = prior_data

        # first gameweek a player plays minutes > 0, until then we use ingest_range
        self._first_gw: dict[str, str] = {}
        
        # cache data in as dict, so lookup cheaper than pandas .loc
        self.player_cache: dict[int, dict[str, dict[str, float]]] = {}

        # cache fixture info for whole season, gw=0 is padded index for priors, all entries but team_code are 0
        self.fixture_cache: dict[int, dict[int, dict[str, int]]] = {0: self._build_prior_fixtures()} 
        self._init_fixture_cache()

    #================================================
    # Initialise Helpers
    #================================================

    def _build_prior_fixtures(self) -> dict[str, int]:
        """Build blank fixture row for prior timesteps; team_code keeps its real value."""
        prior_fixture_row = {}
        for code in self.features._spec_by_name["team_code"].categories:
            prior_fixture_row[code] = {
                feature: (code if feature == "team_code" else -1 if feature == "is_home" else 0)
                for feature in self.features.output_columns_for([DataSource.FIXTURE, DataSource.FIXINGESTER])
            }
        return prior_fixture_row

    def _init_fixture_cache(self) -> "SeasonSequencer":
        """Load fixtures to cache on initialisation. Use _update_team_elo to update future fixture elos."""
        # season runs from gw1-38
        self._ingester.ingest_fixtures_range(1, 38)

        for gw in range(1, 38 + 1):
            self.fixture_cache[gw] = self._ingester.fixtures[gw].to_dict(orient="index")
    
        return self

    #================================================
    # PyTorch Helpers
    # Helps torch.utils.Dataset interface
    #================================================

    @property
    def current_gw(self) -> int:
        """The latest ingested gameweek number."""
        return self._current_gameweek

    def dataset(
        self,
        gw_start: int = 1,
        gw_end: int | None = None,
        player_codes: list[str] | None = None,
    ) -> "FPLDataset":
        """
        Create a Dataset for the specified gameweek range.

        Args:
            gw_start: First target gameweek (inclusive).
            gw_end: Last target gameweek (inclusive).
                Defaults to current_gameweek - target_window_size.
            player_codes: Optional player filter; if None, all players included.

        Returns:
            FPLDataset wrapping this sequencer.
        """
        if gw_end is None:
            gw_end = self._current_gameweek - self._target_window_size
        return FPLDataset(self, gw_start, gw_end, player_codes)

    #================================================
    # Ingestion and State Management
    # Methods for loading gameweeks and advancing state.
    #================================================

    def ingest_range(self, gw_start: int, gw_end: int) -> "SeasonSequencer":
        """
        Bulk-ingest player data for range of gameweeks.

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
            )
            self._prior_data.to_json(self.season_root)

        return self

    @property
    def get_prior(self) -> "PriorData":
        """Returns priors, if calculated."""
        if not self._prior_data: 
            raise RuntimeError("Please ingest to get priors.")

        return self._prior_data
        
    def step(self) -> "SeasonSequencer":
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
        self._ingester.update_future_fixtures(self._current_gameweek)
        self._first_gw = self._ingester.first_gw

        self._cache_gw(self._current_gameweek)


        return self

    #================================================
    # Sequence Construction
    # Methods that build fixed-length player sequences
    # from cached gameweek data and prior estimates.
    #================================================

    def build_player_window(
        self,
        player_team_id: str,
        target_gw: int,
        inference: bool = False,
    ) -> dict[str, np.ndarray | str]:
        """
        Construct a fixed-length sequence of gameweek rows for a single player.

        Builds one unified row per timestep containing all features, then splits
        into continuous and categorical arrays using Features index masks.

        For each timestep, selects either real data or a prior row depending on
        whether the player had appeared in the data by that gameweek.

        Args:
            player_team_id: Composite player-team identifier, e.g. "12345_99".
            target_gw: gameweek to predict, i.e window ends on target - 1
            inference: Whether the sequence is being built for inference.

        Returns:
            Tuple of (continuous, categorical) arrays with shapes
            [window_size, n_continuous] and [window_size, n_categorical].
        """
        # guard against, target_gw being out of ingested data range
        if target_gw <= 0 or (target_gw + self._target_window_size) > 38:
            raise ValueError("Target window must lie within GW1-38")
        if not inference:
            if (target_gw + self._target_window_size) >= self._current_gameweek:
                raise ValueError("Target window contains un-ingested GW, set inference = True, or ingest.")
                
        # static categorical codes, constant across timesteps
        position = self._player_meta[player_team_id]["position"]
        team_code = self._player_meta[player_team_id]["team_code"]
        position_idx = self._player_meta[player_team_id]["position"]
        first_gw = self._first_gw[player_team_id]

        # gw_window is a list of gw numbers, for whoms data we fetch. padded with 0s for prior rows
        gw_window, n_pad = self._build_window(target_gw, first_gw)

        # pre-allocate prior row, to search lookup for multiple weeks
        prior_row = self._build_prior_row(n_pad, player_team_id, team_code, position)

        # input_data: input for model, players stats for each feature, over window gws
        input_data = self._build_input_window(
                player_team_id,
                team_code,
                position_idx,
                prior_row,
                gw_window,
        )
        # given as a context vector to mlp
        future_fixtures = self._build_future_fixtures_window(
            team_code,
            target_gw,
        )
        # data that out model aims to predict
        target_data = self._build_target_window(
            player_team_id,
            target_gw,
            "points",
            inference,
        )

        # split unified array into continuous and categorical using index masks
        full = np.array(input_data, dtype=np.float32)

        x_continuous = full[:, self.features.continuous_indices].astype(np.float32).round(3)
        x_categorical = full[:, self.features.categorical_indices].astype(np.int64)
        x_future_fixtures = np.array(future_fixtures, dtype=np.int64)

        y_predict = np.array(target_data, dtype=np.float32)

        return {
            "player_team_id": player_team_id,
            "x_continuous": x_continuous, 
            "x_categorical": x_categorical, 
            "x_future_fixtures": x_future_fixtures,
            "input_window_length": self._window_size,
            "y": y_predict,
            "target_gw": target_gw,
            "target_window_length": self._target_window_size,

        }

    #================================================
    # Private Helpers
    #================================================

    def _cache_gw(self, gw: int) -> "SeasonSequencer":
        """Convert the ingested DataFrame for a gameweek to a nested dict for fast lookup."""
        self.player_cache[gw] = self._ingester.player_gw_stats[gw].to_dict(orient="index")

        return self

    def _build_window(self, target_gw: int, first_gw: int) -> tuple[list[int], int]:
        """Build the list of gameweeks in the sliding window ending before target_gw."""
        # window must always start at seq_start > 0
        seq_start = max(1, target_gw - self._window_size)

        window = list(range(seq_start, target_gw))

        # filter any gameweeks before the player's first appearance
        # if player hasn't appeared yet first_gw = 39 so gw_window = []
        gw_window = [i for i in window if i >= first_gw]

        # pad to window_size with 0s (prior rows) at the front
        n_pad = self._window_size - len(gw_window)
        gw_window[:0] = [0] * n_pad

        return gw_window, n_pad

    def _build_prior_row(self, n_pad: int, player_team_id: str, team_code: int, position: int) -> list[float | int]:
        """Pre-fetch prior row if any padding needed"""
        if n_pad != 0:
            prior_row = self._get_prior(player_team_id, str(team_code), str(position))
            prior_row.update(self.fixture_cache[0][team_code])
        else:
            prior_row = None

        return prior_row

    def _get_prior(self, player_team_id: str, team_code: str, position: str) -> dict[str, float]:
        """Look up the best available prior for a player, falling back through the hierarchy."""
        if self._prior_data is None:
            raise RuntimeError("No prior data available, call ingest_range() first.")

        pos_team = "_".join([position, team_code])       # NOTE: order is convention

        # lookup player in prior hierarchy
        # load dict as shallow copy, so we dont mutate
        if player_team_id in self._prior_data.individual:
            prior_row = {**self._prior_data.individual[player_team_id]}

        elif pos_team in self._prior_data.position_team:
            prior_row = {**self._prior_data.position_team[pos_team]}

        elif position in self._prior_data.position:
            prior_row = {**self._prior_data.position[position]}

        else:
            prior_row = {**self._prior_data.league["league"]}

        # add row to denote prior, "data_age" is sentinel to be populated later
        prior_row["is_prior"], prior_row["data_age"] = 1, 0.0
        return prior_row

    @staticmethod
    def _minutes_lookup(prior_row: dict[str, float]) -> float:
        """Return a scaled minutes estimate for a player with no recorded data; currently a stub returning 0.0."""
        return 0.0
        
    def _get_real_row(self, player_team_code: str, gw: int) -> dict[str, float]:
        """Retrieve a player's cached stats for a given gameweek."""
        # player lookup in cache
        real_row = {**self.player_cache[gw][player_team_code]}

        # add row to denote prior, "data_age" is sentinel
        real_row["is_prior"], real_row["data_age"] = 0, 0.0
        return real_row

    def _build_input_window(
        self, 
        player_team_id: str, 
        team_code: int, 
        position_idx: int, 
        prior_row: list[float], 
        gw_window: list[int]
    ) -> list[float]:
        """Builds window of input (to model) features"""
        output_columns = self.features.output_columns
        unified_rows = []

        for i, gw in enumerate(gw_window):
            if gw == 0:
                prior_row["data_age"] = self._window_size - i
                row = dict(prior_row)
            else:
                row = self._get_real_row(player_team_id, gw)
                row.update(self.fixture_cache[gw][team_code])

            # categorical codes from player meta and fixture data
            row["position"] = position_idx
            row["team_code"] = team_code

            unified_rows.append([row[col] for col in output_columns])

        return unified_rows

    def _build_future_fixtures_window(self, team_code: int, target_gw: int) -> list[int]:
        """Builds a window of a teams future fixtures."""
        target_window = list(range(target_gw,(target_gw + self._target_window_size)))

        future_fixture = []
        for gw in target_window:
            data_list = list(self.fixture_cache[gw][team_code].values())
            future_fixture.append(data_list)

        return future_fixture

    def _build_target_window(self, player_team_id: str, target_gw: int, target_feature: str, inference: bool) -> list[float]:
        """Builds window of target feature"""
        target_window = list(range(target_gw,(target_gw + self._target_window_size)))

        target_data = []
        for gw in target_window:
            if not inference:
                target_data.append(self.player_cache[gw][player_team_id][target_feature])
            else:
                target_data.append(0.0)

        return target_data


class FPLDataset(Dataset):

    def __init__(
        self, 
        sequencer: SeasonSequencer, 
        gw_start: int, 
        gw_end: int, 
        player_team_id: list[str] | None = None,
        inference: bool = False
    ):
        self.sequencer = sequencer
        self._inference = inference 

        # --- validate gameweek range ---
        if gw_start < 1:
            raise ValueError(f"gw_start must be >= 1, got {gw_start}.")
        if gw_end < gw_start:
            raise ValueError(
                f"gw_end ({gw_end}) must be >= gw_start ({gw_start})."
            )

        target_end = gw_end + sequencer._target_window_size
        if target_end > 38:
            raise ValueError(
                f"Target window extends to GW{target_end}, beyond GW38."
            )
        if not inference and target_end > sequencer.current_gw:
            raise ValueError(
                f"Target window extends to GW{target_end} but only "
                f"GW1-{sequencer.current_gw} are ingested. "
                "Set inference=True or ingest more gameweeks."
            )

        # --- determine player set ---

        if player_team_id is not None:
            known = set(sequencer._player_meta.keys())
            unknown = set(player_team_id) - known
            if unknown:
                raise ValueError(
                    f"Unknown player codes: {unknown}"
                )
            players = list(player_team_id)
        else:
            players = list(sequencer._player_meta.keys())

        # --- build flat sample index ---
        # Order: all players for GW1, then all for GW2, etc.
        self._sample_index: list[tuple[str, int]] = [
            (player_id, gw)
            for gw in range(gw_start, gw_end + 1)
            for player_id in players
        ]

        logger.info(
            "FPLDataset created: %d players x GW%d-%d = %d samples",
            len(players), gw_start, gw_end, len(self._sample_index),
        )

    #====================================================================
    # Dataset Rewrites
    # allows torch to handle interface between sequencer and model
    #====================================================================

    def __len__(self): 
        """Return the total number of (player, gameweek) samples."""
        return len(self._sample_index)

    def __getitem__(self, idx: int):
        """
        Retrieve a single training sample as a dictionary of tensors.

        Args:
            idx: Integer index into the sample index.

        Returns:
            Dictionary with keys:
                x_continuous: float32 tensor, shape [T, F_cont].
                x_categorical: long tensor, shape [T, C].
                x_future_fixtures: long tensor, shape [K, fix_features].
                y: float32 tensor, shape [target_window_size].
        """
        player_team_code, target_gw = self.sample_index[idx]

        sample = self.sequencer.build_player_window(
            player_team_code, 
            target_gw,
            inference=self._inference,
            )
        
        return {
            "x_continuous": torch.tensor(
                sample["x_continuous"], dtype=torch.float32,
            ),
            "x_categorical": torch.tensor(
                sample["x_categorical"], dtype=torch.long,
            ),
            "x_future_fixtures": torch.tensor(
                sample["x_future_fixtures"], dtype=torch.long,
            ),
            "y": torch.tensor(
                sample["y"], dtype=torch.float32,
            ),
        }

    #================================================
    # Inspection Helpers
    #================================================

    @property
    def sample_index(self) -> list[tuple[str, int]]:
        """Copy of the (player_team_id, target_gw) index for inspection."""
        return list(self._sample_index)