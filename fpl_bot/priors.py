import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
import json
from .features import Features, FeatureSpec, FeatureType
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriorData:
    """
    Container for the four hierarchy levels.
    Serialisable to/from JSON.
    """
    league: dict[str, dict[str, float]]         #
    position: dict[str, dict[str, float]]       #   
    position_team: dict[str, dict[str, float]]  #
    individual: dict[str, dict[str, float]]     # keys are str because JSON requires string keys
    meta_data: dict                             #

    # --- Public Functions & Methods ---
    
    def to_json(self, path: str):
        with open(path + f"priors.json", "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, path: str, filename: str) -> 'PriorData':
        with open(path + filename, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_data(
        cls,
        player_data: dict[int, pd.DataFrame],
        cumulative: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,
        cum_rev_map: dict[str, str],
        min_mins: float = 450.0,
    ) -> 'PriorData':
        """Convenience constructor - delegates to PriorComputer."""
        return PriorComputer(player_data, cumulative, player_meta, cum_rev_map, min_mins).compute()


class PriorComputer:
    """
    Computes hierarchical per-90 priors from ingester output.
    Data-agnostic, works with any number of gameweeks.
    Uses max(cumulative.keys()) as the latest available snapshot.

    Args:
        cumulative:   from ingester, raw cumulative counts per (player, gw)
        player_data:  from ingester, per-GW features and per-90 rates
        player_meta:  DataFrame with [player_code, position, team_code]
        min_mins:     minimum cumulative minutes for a player to qualify
                      for individual-level priors (default 450 ≈ 5 full matches)
        cum_rev_map:  maps cum_ columns to _per_90 columns for consistency with ingester
    """

    def __init__(
        self,
        player_data: dict[int, pd.DataFrame],
        cumulative: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,
        cum_rev_map: dict[str, str],
        min_mins: float = 450.0,
    ):
        self.cumulative = cumulative
        self.player_data = player_data
        
        self.cum_rev_map = cum_rev_map      
        self.min_mins = min_mins
        self._validate_input()

        self.latest_gw = max(cumulative.keys())
        self.snapshot_cols = [                                  # snapshot featues
            col for col in player_data[self.latest_gw].columns 
            if "per_90" not in col and "minutes" not in col
            ]
        self.per_90_cols = [                                    # per_90 features
            col for col in player_data[self.latest_gw].columns 
            if col not in self.snapshot_cols
        ]

        if player_meta.index.name != "player_team_id":
            self.player_meta = player_meta.set_index("player_team_id")
        else: 
            self.player_meta = player_meta

        # populated during compute()
        self.totals: pd.DataFrame | None = None

    # --- Validation ---

    def _validate_input(self):
        if self.cumulative.keys() != self.player_data.keys():
            raise RuntimeError("Inconsistent no. of GWs entered into compute priors")
        for key in self.cumulative.keys():
            if (self.cumulative[key].index.name != "player_team_id") or (self.player_data[key].index.name != "player_team_id"):
                raise TypeError("compute_priors() requires index is player_team_id for all df")

    # --- Private helpers ---

    @staticmethod
    def _coerce_categorical_cols(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """
        WARNING: mutates input df, provide copy to avoid 
        Makes cat_cols of df into categorical type,
        when called with .groupby(observed=False) forces calculation of all permutations even if no contributors
        """

        for col in cat_cols:
            # validate
            if not col in df.columns:
                raise ValueError("All cat_cols must be columns in DataFrame")
                
            df[col]= df[col].astype("category")
        
        return df

    def _per_90_calculation(self, cum_df: pd.DataFrame) -> pd.DataFrame:
        """
        calculates per_90 rate from cumulative df,
        also maps cumulative columns names to per_90 columns with cum_rev_map
        """
        return (
            cum_df
            .drop("cum_minutes", axis=1)
            .div(cum_df["cum_minutes"] / 90, axis=0)
            .replace([np.inf, -np.inf, np.nan], 0.0)
            .rename(columns=self.cum_rev_map)
        )

    @staticmethod
    def _variance_of_ratio(df: pd.DataFrame, group: dict, numerator: str, denominator: str) -> pd.Series:
        """
        Calulates variance for ratio of two ratio_cols.
        NOTE: second col is numerator
        WARNING: only call if group not {"level": 0}
        """
        ratio = df[group["by"] + [numerator, denominator]]
        ratio["ratio"]= df[numerator] / df[denominator]
        
        return ratio.groupby(**group)["ratio"].std().fillna(0.0)

    @staticmethod
    def _output_df_to_dict(priors: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        convert df of prior computation results to dict, 
        any dict keys must be str for serialization
        """
        result = {}
        for idx, row in priors.iterrows():
            # if groupby was on multiple cols, index is tuple
            key = "_".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
            result[key] = row.to_dict()

        return result

    def _sum_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sums weighted columns by summing (feature * minutes) per gw
        """
        # for snapshot columns we want to start weighted average
        return (    
            df[self.snapshot_cols]
            .multiply(df["minutes"], axis=0)
            .groupby(level=0)
            .sum()
        )

    def _normalise_weighted_sums(self, df: pd.DataFrame, group: dict) -> pd.DataFrame:    
        """
        WARNING: mutates input, provide copy to avoid
        Leaving group_cols empty, groups by index
        """
        # calculate weighted (minutes) average per group, group first for efficency
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
        """
        returns one combined df, from input of player_map, player_data and cumulative
        - stacks player data to allow for weighted sum of snapshot features
        - takes most recent entry to cumulative dict
        """
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
        """
        Compute priors for given level, defined by group cols.
        Args: 
            - group_cols: defines level, if None groups by index ("player_team_id") 
            - input_df:   reads df from state or can receive df as parameter
        """
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

        # rate columns
        rate_cols = [col for col in df.columns if "cum_" in col]

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
            df[rate_cols + group_cols]
            .groupby(**group)
            .sum()
        )
        per_90_df = self._per_90_calculation(cum_df)

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
        combined = pd.concat([per_90_df, snapshots, cum_df[["minutes", "mins_over_featured_var"]]], axis=1)

        # output dict
        return self._output_df_to_dict(combined)

    # --- Public Functions ---

    def compute(self) -> PriorData:
        """Main entry point — orchestrates the full pipeline."""
        self.totals = self._extract_totals()

        # calculate position, position_team and individual
        position = self._compute_level(group_cols=["position"])
        pos_team = self._compute_level(group_cols=["position", "team_code"])
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
                "snapshot_cols": self.snapshot_cols,
                "per_90_cols": self.per_90_cols,
            }
        )

