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
        with open(path + f"GW{self.meta_data['latest_gw']}_priors.json", "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, path: str, filename: str) -> 'PriorData':
        with open(path + filename, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_data(
        cls,
        cumulative: dict[int, pd.DataFrame],
        player_data: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,

        min_mins: float = 450.0,
    ) -> 'PriorData':
        """Convenience constructor - delegates to PriorComputer."""
        return PriorComputer(cumulative, player_data, player_meta, min_mins).compute()


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
        cumulative: dict[int, pd.DataFrame],
        player_data: dict[int, pd.DataFrame],
        player_meta: pd.DataFrame,
        cum_rev_map: dict[str, str],
        min_mins: float = 450.0,
    ):
        self.cumulative = cumulative
        self.player_data = player_data
        
        self.cum_rev_map = cum_rev_map      
        self.min_mins = min_mins
        self.latest_gw = max(cumulative.keys())
        self.weighted_cols = ["points", "value"]

        if "player_id" in player_meta.columns:
            self.player_meta = player_meta.set_index("player_code")
        else: 
            self.player_meta = player_meta

        # populated during compute()
        self.totals: pd.DataFrame | None = None

        self._validate_input()

    # --- Validation ---

    def _validate_input(self):
        if self.cumulative.keys() != self.player_data.keys():
            raise RuntimeError("Inconsistent no. of GWs entered into compute priors")
        for key in self.cumulative.keys():
            if (self.cumulative[key].index.name != "player_code") or (self.player_data[key].index.name != "player_code"):
                raise TypeError("compute_priors() requires index is player_code for all df")

    # --- Private helpers ---

    def _extract_totals(self) -> pd.DataFrame:
        # stack player_data into one df
        player_stacked = pd.concat([df[self.weighted_cols + ["minutes"]] for df in self.player_data.values()], axis=0)

        # for snapshot columns we want to start weighted average
        weighted_snapshots = (    
            player_stacked[self.weighted_cols]
            .multiply(player_stacked["minutes"], axis=0)
            .groupby(level=0)
            .sum()
        )

        # for cumulative data we just want most recent count
        cumulative = self.cumulative[max(self.cumulative.keys())]

        # combine 
        output = (
            pd.concat([self.player_meta, cumulative, weighted_snapshots], axis=1)
            .fillna(0)
        )
        return output

    def _compute_league(self) -> dict[str, dict[str, float]]:
        # find league averages:
        league = {}
        sumation = (
            self.totals
            .drop(["position", "team"], axis=1)
            .sum()
        )
        rates = sumation.drop("cum_minutes").div(sumation["cum_minutes"] / 90)
        rates = rates.replace([np.inf, -np.inf, np.nan], 0.0)
        rates["cum_minutes"] = sumation["cum_minutes"]
        
        # rename to per_90 column names
        rates = rates.rename(columns=self.cum_rev_map)

        league["league"] = rates.to_dict()

        return league

    def _compute_level(self, group_cols: list[str]) -> dict[str, dict[str, float]]:
        # copy dataframe, so no mutation
        df = self.totals.copy()

        # rate columns
        rate_cols = [col for col in df.columns if "cum_" in col]

        # forces pandas to calculate each permutation, even if no data
        for col in group_cols:
            df[col] = df[col].astype("category")

        # load to dict for ease
        group = {"by": group_cols, "observed": False}

        # find per_90 averages for groups
        sumation = (
            df[rate_cols + group_cols]
            .groupby(**group)
            .sum()
        )
        rates = (
            sumation
            .pipe(lambda d: d.drop("cum_minutes", axis=1).div((d["cum_minutes"] / 90), axis=0))
            .replace([np.inf, -np.inf, np.nan], 0.0) 
        ) 
        # find weighted averages of snapshot features
        snapshots = self._weighted_average(df[self.weighted_cols + ["cum_minutes"] + group_cols], group_cols)
        
        # join snapshot and rate calculations
        combined = rates.join(snapshots).join(sumation["cum_minutes"])

        # rename to per_90 column names
        rates = rates.rename(columns=self.cum_rev_map)

        # populate results dict, idx is tuple for multiple group_cols, str for single
        result = {}
        for idx, row in combined.iterrows():
            key = "_".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
            result[key] = row.to_dict()
        
        return result

    def _compute_individual(self) -> dict[str, dict[str, float]]:
        df = self.totals

        # rate columns
        rate_cols = [col for col in df.columns if "cum_" in col]

        #filter minutes
        filtered = df[df["cum_minutes"] > self.min_mins]

        # calculate per 90 rates per player, after filtering minutes
        rates = (
            filtered[rate_cols]
            .pipe(lambda d: d[d["cum_minutes"] > self.min_mins])
            .pipe(lambda d: d.drop("cum_minutes", axis=1).div((d["cum_minutes"] / 90), axis=0))  
            .replace([np.inf, -np.inf, np.nan], 0.0)     
        )
        
        # rename to per_90 column names
        rates = rates.rename(columns=self.cum_rev_map)

        # find weighted averages of snapshot features, only over players with minutes > 450
        snapshots = self._weighted_average(filtered[self.weighted_cols + ["cum_minutes"]], [])
        
        # join snapshot and rate calculations
        combined = rates.join(snapshots).join(filtered["cum_minutes"])

        # construct dict
        result = {}
        for idx, row in combined.iterrows():
            result[str(idx)] = row.to_dict()

        return result

    @staticmethod
    def _weighted_average(
        df: pd.DataFrame,
        group_cols: list[str],
    ):
        """
        Leaving group_cols empty, groups by index
        WARNING: this class calulates weights, after sum over (feature * weight)
        """
        # copy dataframe, so no mutation
        df = df.copy()

        # forces pandas to calculate each permutation, even if no data 
        for col in group_cols:
            df[col] = df[col].copy().astype("category")

        # load to dict for ease
        group = {"by": group_cols, "observed": False} if group_cols else {"level": 0}

        # calculate weighted (minutes) average per group, group first for efficency
        grouped = df.groupby(**group).sum()
        return (
            grouped
            .div(grouped["cum_minutes"], axis=0)
            .replace([np.inf, -np.inf, np.nan], 0.0)
            .drop("cum_minutes", axis=1)
        )

    # --- Public Functions ---

    def compute(self) -> PriorData:
        """Main entry point — orchestrates the full pipeline."""
        self.totals = self._extract_totals()

        # calculate league, position, position_team and individual
        league = self._compute_league()
        position = self._compute_level(group_cols=["position"])
        pos_team = self._compute_level(group_cols=["position", "team"])
        players = self._compute_individual()

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
                "weighted_cols": [cols for cols in self.player_data[max(self.cumulative.keys())].columns if "per_90" not in cols],
                "per_90_cols": [cols for cols in self.player_data[max(self.cumulative.keys())].columns if "per_90" in cols],
            }
        )

