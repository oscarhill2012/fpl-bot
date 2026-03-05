import torch 
import pandas as pd
import numpy as np
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Callable
import math 
from .features import Features, FeatureSpec, FeatureType
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_totals(
    cumulative: dict[int, pd.DataFrame],
    player_data: dict[int, pd.DataFrame],
    player_meta: pd.DataFrame,
):
    # check consistent gws
    if not cumulative.keys() == player_data.keys():
        raise ValueError("Inconsistent GW range fed into compute_priors()")

    # identify most recent data
    latest_gw = max(cumulative.keys())

    # average non-cumulative features over season
    features = [cols for cols in player_data[latest_gw].columns if "per_90" not in cols]
    averaged = (
        pd.concat(df[features] for df in player_data.values())
        .pipe(lambda d: d[d["minutes"] > 0])
        .groupyby(level=0)
        .mean()
    )

    # combine 
    output = (
        pd.concat([player_meta, cumulative[latest_gw], averaged], axis=1)
        .fillna(0)
    )

    return output

def _compute_level(
    df: pd.DataFrame,
    group_cols: list[str] | None,
) -> dict[str, dict[str, float]]:

    # forces pandas to calculate each permutation, even if no data
    for col in group_cols:
        df[col] = df[col].astype("category")

    # find per_90 averages for groups
    grouped = (
        df
        .groupby(group_cols, observed=False)
        .sum()
        .pipe(lambda d: d.drop(columns="minutes").div((d["minutes"] / 90), axis=0))
        .replace([np.inf, -np.inf, np.nan], 0)   
    ) 

    # populate results dict, idx is tuple for multiple group_cols, str for single
    result = {}
    for idx, row in grouped.iterrows():
        key = "_".join(str(i) for i in idx) if isinstance(idx, tuple) else str(idx)
        result[key] = row.to_dict()
    
    return result

def _compute_individual(
    df: pd.DataFrame,
    min_mins: float,
) -> dict[str, dict[str, float]]:

    # filter minutes, thus no fear of div by 0
    df = (
        df[df["minutes"] > min_mins]
        .drop(["position", "team"])
        .pipe(lambda d: d.drop(columns="minutes").div((d["minutes"] / 90), axis=0))        
    )
    # construct dict
    result = {}
    for idx, row in df.iterrow():
        result[idx] = row.to_dict()

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
    # set index as player_code
    player_meta = player_meta.index("player_code")

    # calculate totals
    totals = _extract_totals(cumulative, player_data, player_meta)

    # calculate league, position, position_team and individual
    position = _compute_level(totals, group_cols=["position"])
    pos_team = _compute_level(totals, group_cols=["position", "team"])
    players = _compute_individual(totals, min_mins)
    
    # find league averages:
    league = {}
    league["league"] = (
        totals
        .drop(["position", "minutes"])
        .sum()
        .pipe(lambda d: d.drop(columns="minutes").div((d["minutes"] / 90), axis=0))
        .to_dict()
    )

    return PriorData(
        league,
        position=position,
        position_team=pos_team,
        individual=players,
        meta_data = {
            "latest_gw": max(cumulative.keys()),
            "min_minutes": min_mins,
            "n_players_tot": len(player_meta),
            "n_player_mins_req": len(players),
            "snapshot_cols": [cols for cols in player_data[max(cumulative.keys())].columns if "per_90" not in cols],
            "per_90_cols": [cols for cols in player_data[max(cumulative.keys())].columns if "per_90" in cols],
        }

    )

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
        pass
    @classmethod
    def from_json(cls, path: str) -> 'PriorData':
        pass