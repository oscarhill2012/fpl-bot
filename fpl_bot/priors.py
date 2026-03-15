from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from .features import Features, FeatureSpec, FeatureType
from .player_team_index import player_team_index
import logging
logger = logging.getLogger(__name__)

@dataclass
class PriorData:
    """
    Container for the four-level prior hierarchy.

    Serialisable to and from JSON.
    """
    league: dict[str, dict[str, float]]         #
    position: dict[str, dict[str, float]]       #
    position_team: dict[str, dict[str, float]]  #
    individual: dict[str, dict[str, float]]     # keys are str because JSON requires string keys
    meta_data: dict                             #

    #================================================
    # Public Functions
    #================================================

    def to_json(self, path: str) -> None:
        """
        Serialise this PriorData instance to a JSON file.

        Args:
            path: Directory path; the file is written as path/priors.json.
        """
        with open(path + f"/priors.json", "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, path: str, filename: str) -> "PriorData":
        """
        Load a PriorData instance from a JSON file.

        Args:
            path: Directory containing the file.
            filename: Filename to read, e.g. "/priors.json".

        Returns:
            Deserialised PriorData instance.
        """
        with open(path + filename, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_data(
        cls,
        features: Features,
        player_data: dict[int, pd.DataFrame],
        cumulative: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,
        min_mins: float = 450.0,
    ) -> "PriorData":
        """
        Convenience constructor — delegates to PriorComputer.

        Args:
            features: Global feature registry.
            player_data: Per-GW features and per-90 rates from the ingester.
            cumulative: Raw cumulative counts per player per gameweek.
            player_meta: DataFrame with columns [player_code, position, team_code].
            min_mins: Minimum cumulative minutes to qualify for individual priors.

        Returns:
            Computed PriorData instance.
        """
        return PriorComputer(features, player_data, cumulative, player_meta, min_mins).compute()


class PriorComputer:
    """
    Computes hierarchical per-90 priors from ingester output.

    Data-agnostic; works with any number of gameweeks.
    Uses max(cumulative.keys()) as the latest available snapshot.

    Args:
        features:     Global feature registry.
        cumulative:   From ingester; raw cumulative counts per (player, gw).
        player_data:  From ingester; per-GW features and per-90 rates.
        player_meta:  DataFrame with [player_code, position, team_code].
        min_mins:     Minimum cumulative minutes for a player to qualify
                      for individual-level priors (default 450 ≈ 5 full matches).
        cum_rev_map:  Maps cum_ columns to _per_90 columns for consistency with ingester.
    """

    def __init__(
        self,
        features: Features,
        player_data: dict[int, pd.DataFrame],
        cumulative: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,
        min_mins: float = 450.0,
    ):
        """
        Initialise the PriorComputer.

        Args:
            features: Global feature registry.
            player_data: Per-GW features and per-90 rates from the ingester.
            cumulative: Raw cumulative counts per player per gameweek, from the ingester.
            player_meta: DataFrame with columns [player_code, position, team_code].
            min_mins: Minimum cumulative minutes a player must have played to qualify
                for individual-level priors. Default is 450 (approximately 5 full matches).
        """
        self.cumulative = cumulative
        self.player_data = player_data
        self.min_mins = min_mins
        self.latest_gw = max(cumulative.keys())

        self._validate_input()

        self.features = features
        self.cumulative_cols = features.cumulative_columns
        self.per_90_cols = features.per_90_columns
        self.snapshot_cols = features.snapshot_columns
        self.cum_rev_map = features.inv_cumulative_map

        if player_meta.index.name != "player_team_id":
            player_meta["player_team_id"] = player_team_index(player_meta)
            self.player_meta = player_meta.set_index("player_team_id")
        else:
            self.player_meta = player_meta

        # populated during compute()
        self.totals: pd.DataFrame | None = None

    #================================================
    # Public Functions
    #================================================

    def compute(self) -> "PriorData":
        """
        Orchestrate the full prior computation pipeline.

        Returns:
            PriorData containing priors at all four hierarchy levels.
        """
        self.totals = self._extract_totals()

        # calculate position, position_team and individual
        position = self._compute_level(group_cols=["position"])
        pos_team = self._compute_level(group_cols=["position", "team_code"])    # NOTE: position team ordering is convention
        players = self._compute_level(minutes_threshold=self.min_mins)

        # add league groupby col and calculate league prior
        league_df = self.totals.copy()
        league_df["league"] = "league"
        # no minutes threshold as we want all contributions for league wide calc
        league = self._compute_level(group_cols=["league"], input_df=league_df)

        return PriorData(
            league=league,
            position=position,
            position_team=pos_team,
            individual=players,
            meta_data={
                "latest_gw": self.latest_gw,
                "min_minutes": self.min_mins,
                "n_players_tot": len(self.player_meta),
                "n_player_mins_req": len(players),
                "snapshot_cols": self.features.snapshot_columns,
                "per_90_cols": self.features.per_90_columns,
            }
        )

    #================================================
    # Private Helpers
    #================================================

    def _validate_input(self):
        """Validate that cumulative and player_data dicts match gameweeks and index names."""
        if self.cumulative.keys() != self.player_data.keys():
            raise RuntimeError("Inconsistent no. of GWs entered into compute priors")
        for key in self.cumulative.keys():
            if (self.cumulative[key].index.name != "player_team_id") or (self.player_data[key].index.name != "player_team_id"):
                raise TypeError("compute_priors() requires index is player_team_id for all df")

    @staticmethod
    def _coerce_categorical_cols(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """Cast specified columns to categorical dtype for groupby with observed=False."""
        for col in cat_cols:
            # validate
            if not col in df.columns:
                raise ValueError("All cat_cols must be columns in DataFrame")

            df[col]= df[col].astype("category")

        return df

    def _per_90_calculation(self, cum_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-90 rates from cumulative totals, renaming columns via cum_rev_map."""
        return (
            cum_df[self.per_90_cols]
            .drop("cum_minutes", axis=1)
            .div(cum_df["cum_minutes"] / 90, axis=0)
            .replace([np.inf, -np.inf, np.nan], 0.0)
            .rename(columns=self.cum_rev_map)
        )

    @staticmethod
    def _variance_of_ratio(df: pd.DataFrame, group: dict, numerator: str, denominator: str) -> pd.Series:
        """Calculate the per-group standard deviation of the ratio numerator/denominator."""
        ratio = df[group["by"] + [numerator, denominator]].copy()
        ratio["ratio"]= df[numerator] / df[denominator]

        return ratio.groupby(**group)["ratio"].std().fillna(0.0)

    @staticmethod
    def _output_df_to_dict(priors: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Convert a prior result DataFrame to a string-keyed dict for JSON serialisation."""
        result = {}
        for idx, row in priors.iterrows():
            # if groupby was on multiple cols, index is tuple
            key = "_".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
            result[key] = row.to_dict()

        return result

    def _sum_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sum snapshot features weighted by minutes, grouped by player."""
        # for snapshot columns we want to start weighted average
        return (
            df[self.snapshot_cols]
            .multiply(df["minutes"], axis=0)
            .groupby(level=0)
            .sum()
        )

    def _normalise_weighted_sums(self, df: pd.DataFrame, group: dict) -> pd.DataFrame:
        """Divide grouped weighted sums by cumulative minutes to produce weighted averages."""
        # calculate weighted (minutes) average per group, group first for efficiency
        grouped = (
            df
            .groupby(**group)
            .sum()
        )
        return (
            grouped[self.snapshot_cols]
            .div(grouped["cum_minutes"], axis=0)
            .replace([np.inf, -np.inf, np.nan], 0.0)
        )

    def _extract_totals(self) -> pd.DataFrame:
        """Build a combined DataFrame of cumulative and snapshot totals for all players."""
        # stack player_data into one df
        player_stacked = pd.concat([df[self.snapshot_cols + ["minutes"]] for df in self.player_data.values()], axis=0)

        snapshot_features = self._sum_weighted_features(player_stacked)

        # for cumulative data we just want most recent count
        cumulative = self.cumulative[max(self.cumulative.keys())]

        # combine
        output = (
            pd.concat([self.player_meta, cumulative, snapshot_features], axis=1)
            .fillna(0.0)
        )
        return output

    def _compute_level(self, group_cols: list[str] | None = None, input_df: pd.DataFrame | None = None, minutes_threshold: float = 0.0) -> dict[str, dict[str, float]]:
        """Compute priors for a given hierarchy level, defined by group_cols."""
        # copy dataframe, so no mutation
        if input_df is None:
            df = self.totals.copy()
        else:
            df = input_df.copy()

        # avoid mutable default argument
        if not group_cols:
            group_cols = []

        # filter above threshold minutes and featured (games involved in)
        df = df[df["cum_minutes"] > minutes_threshold]
        df = df[df["cum_featured"] > 0]

        if group_cols:
            # forces pandas to calculate each permutation, even if no data
            df = self._coerce_categorical_cols(df, group_cols)
            # load to dict for ease
            group = {"by": group_cols, "observed": False}
        else:
            group = {"level": 0}

        # find variance of (cum_minutes / cum_features), as measure of priors predictive accuracy
        if group_cols:
            ratio_var = self._variance_of_ratio(df, group, "cum_minutes", "cum_featured")
        else:
            ratio_var = 0.0

        # find per_90 averages for groups
        cum_df = (
            df
            .groupby(**group)
            .sum()
        )
        per_90 = self._per_90_calculation(cum_df)

        # find weighted averages of snapshot features
        snapshots = self._normalise_weighted_sums(df, group)

        # add average minutes per time featured in game, as an uneducated guess at minutes,
        # this will be refined with MinutesEstimator
        cum_df["minutes"] = (
            cum_df["cum_minutes"]
            .div(cum_df["cum_featured"], axis=0)
            .replace([np.inf, -np.inf, np.nan], 0.0)
        )
        cum_df["mins_over_featured_var"] = ratio_var

        # join snapshot and rate calculations
        combined = pd.concat([per_90, snapshots, cum_df[["minutes", "mins_over_featured_var"]]], axis=1)

        # ensure combined has columns ordered by convention
        combined = combined[self.features.pre_sequencer_columns]

        # output dict
        return self._output_df_to_dict(combined)
