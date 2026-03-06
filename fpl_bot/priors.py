import pandas as pd
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable
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

    def to_json(self, path: str):
        with open(path + f"GW{self.meta_data['latest_gw']}_priors.json", "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_json(cls, path: str, filename: str) -> 'PriorData':
        with open(path + filename, "r") as f:
            data = json.load(f)
        return cls(**data)

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

    # calculate weighted (minutes) average per group
    return (
        df
        .groupby(**group).sum()
        .div(df[["cum_minutes"]].groupby(**group).sum(), axis=0)
        .replace([np.inf, -np.inf, np.nan], 0)
        .drop("cum_minutes", axis=1)
    )


def _extract_totals(
    cumulative_dfs: dict[str, pd.DataFrame],
    player_data: dict[str, pd.DataFrame],
    player_meta: pd.DataFrame,
    weighted_cols: list,
):  
    # stack player_data into one df
    player_stacked = pd.concat([df[weighted_cols + ["minutes"]] for df in player_data.values()], axis=0)

    # for snapshot columns we want to start weighted average
    weighted_snapshots = (    
        player_stacked[weighted_cols]
        .multiply(player_stacked["minutes"], axis=0)
        .groupby(level=0)
        .sum()
    )

    # for cumulative data we just want most recent count
    cumulative = cumulative_dfs[max(cumulative_dfs.keys())]

    # combine 
    output = (
        pd.concat([player_meta, cumulative, weighted_snapshots], axis=1)
        .fillna(0)
    )
    return output


def _compute_level(
    df: pd.DataFrame,
    weighted_cols: list,
    group_cols: list[str],
) -> dict[str, dict[str, float]]:
    # copy dataframe, so no mutation
    df = df.copy()

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
        .replace([np.inf, -np.inf, np.nan], 0) 
    ) 
    # find weighted averages of snapshot features
    snapshots = _weighted_average(df[weighted_cols + ["cum_minutes"]], group_cols)
    
    # join snapshot and rate calculations
    combined = rates.join(snapshots).join(sumation["cum_minutes"])

    # populate results dict, idx is tuple for multiple group_cols, str for single
    result = {}
    for idx, row in combined.iterrows():
        key = "_".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
        result[key] = row.to_dict()
    
    return result

def _compute_individual(
    df: pd.DataFrame,
    weighted_cols: list,
    min_mins: float,
    group_cols: list,
) -> dict[str, dict[str, float]]:
    # rate columns
    rate_cols = [col for col in df.columns if "cum_" in col]

    #filter minutes
    filtered = df[df["cum_minutes"] > min_mins]

    # calculate per 90 rates per player, after filtering minutes
    rates = (
        filtered[rate_cols]
        .pipe(lambda d: d[d["cum_minutes"] > min_mins])
        .pipe(lambda d: d.drop("cum_minutes", axis=1).div((d["cum_minutes"] / 90), axis=0))       
    )
    
    # find weighted averages of snapshot features, only over players with minutes > 450
    snapshots = _weighted_average(filtered[weighted_cols + ["cum_minutes"]], group_cols)
    
    # join snapshot and rate calculations
    combined = rates.join(snapshots).join(filtered["cum_minutes"])

    # construct dict
    result = {}
    for idx, row in combined.iterrows():
        result[str(idx)] = row.to_dict()

    return result

def compute_priors(
    cumulative: dict[int, pd.DataFrame],
    player_data: dict[int, pd.DataFrame],
    player_meta: pd.DataFrame,
    min_mins: float = 450.0, 
) -> "PriorData":
    """
    Computes hierarchical per-90 priors from ingester output.
    Data-agnostic — works with any number of gameweeks.
    Uses max(cumulative_dict.keys()) as the latest available snapshot.

    Args:
        cumulative_dict:  from ingester — raw cumulative counts per (player, gw)
        output_dict:      from ingester — per-GW features and per-90 rates
        player_meta:      DataFrame with [player_code, position, team_code]
        min_minutes:      minimum cumulative minutes for a player to qualify
                          for individual-level priors (default 450 ≈ 5 full matches)

    Returns:
        PriorData containing three hierarchy levels + metadata
    """
    # test input:
    if cumulative.keys() != player_data.keys():
        raise RuntimeError("Inconsistent no. of GWs entered into compute priors")
    for key in cumulative.keys():
        if (cumulative[key].index.name != "player_code") or (player_data[key].index.name != "player_code"):
            raise TypeError("compute_priors() requires index is player_code for all df")
        
    # set index as player_code
    player_meta = player_meta.set_index("player_code")
    weighted_cols = ["points", "value"]

    # extract season wide totals and weighted averages
    latest_gw = max(cumulative.keys())
    totals = _extract_totals(cumulative, player_data, player_meta, weighted_cols)

    # calculate league, position, position_team and individual
    position = _compute_level(totals, weighted_cols , group_cols=["position"])
    pos_team = _compute_level(totals, weighted_cols , group_cols=["position", "team"])
    players = _compute_individual(totals, weighted_cols, min_mins, group_cols=[])
    
    # find league averages:
    league = {}
    sumation = (
        totals
        .drop(["position", "team"], axis=1)
        .sum()
    )
    league["league"] = (
        sumation
        .pipe(lambda d: d.drop("cum_minutes").div((d["cum_minutes"] / 90), axis=0))
        .join(sumation["cum_minutes"])
        .to_dict()
    )
    
    return PriorData(
        league=league,
        position=position,
        position_team=pos_team,
        individual=players,
        meta_data = {
            "latest_gw": latest_gw,
            "min_minutes": min_mins,
            "n_players_tot": len(player_meta),
            "n_player_mins_req": len(players),
            "weighted_cols": [cols for cols in player_data[max(cumulative.keys())].columns if "per_90" not in cols],
            "per_90_cols": [cols for cols in player_data[max(cumulative.keys())].columns if "per_90" in cols],
        }
    )


