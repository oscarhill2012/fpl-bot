import torch 
import pandas as pd
import logging
from enum import Enum
import math 
from .features import Features, FeatureSpec, FeatureType
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FCIIngester:
    """
    Ingests one season from FPL-Core-Insights.
    Produces a DataFrame with per-90 rates + per-GW + snapshot features
    per (player_code, gw).
    
    The ingestion logic is identical for both seasons — playermatchstats
    has 54 identical columns, and the playerstats columns we use
    (ICT, BPS, bonus, snapshots) exist in both.

    WARNING:this class is specifically for 24-25 season data loading,
            for some reason, there are no cumulative statistics provided        
    """

    def __init__(
        self,
        file_path: str,
        window: int,
        gw_start: int,
        features: Features,
        canonical_feature: FeatureSpec,
        threshold: float,
        vaastav_map: dict, 
        vaastav_features: list[str],
        sanity_check: bool = False,
    ):
        self.window = window
        self.file_path = file_path
        self.local_id = "player_id"
        self.global_id = "player_code"
        self.features = features                            # Features object
        self.canonical_feature = canonical_feature          # this feature is the denominator of canonicalisation
        self.threshold = threshold                          # this is a threshold for canonical feature, below which samples are ignore
        self._sanity_check = sanity_check                   # this checks certain features against vaastav cumulative tally to ensure we not go crazy
        self.vaastav_features = vaastav_features
        self.vaastav_map = vaastav_map
        self.gw_start = gw_start
        self.global_id_list = []    
        self.players = pd.DataFrame()
        self.cumulative_player_stats = pd.DataFrame()
    
    """
    Build internal functions
    """

    def _move_col_to_front(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        cols = [col] + [c for c in df.columns if c != col]
        return df[cols]

    def _validate_df(self, df: pd.DataFrame):     
        # check all ids exist
        logger.info("Checking IDs")
        actual_ids = set(df[self.local_id])
        max_id = max(actual_ids)
        missing_ids = sorted(set(range(1, max_id + 1)) - actual_ids)

        if missing_ids:
            logger.warning(f"Missing IDs from '{self.local_id}' column: {missing_ids}")
        else:
            logger.info(f"All IDs from 1 to {max_id} present.")

    def _process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        #process df
        return (
            df
            .merge(self.players[[self.local_id, self.global_id]], on=self.local_id, how="right")
            .drop(columns=self.local_id)
            .fillna(0)
            .pipe(self._move_col_to_front, col=self.global_id)
            .reset_index(drop=True)
        )

    """
    Build core functions
    """

    def IngestSeason(self) -> dict[int, pd.DataFrame]:
        # from player, get player_code and team_code and position
        self.players = pd.read_csv(self.file_path + f"/players/players.csv")
        self._validate_df(self.players)
        self.global_id_list = list(self.players[self.global_id].unique())
        
        # from teams, get team strength (working under assumption that all are up to date for end of season)
        teams = pd.read_csv(self.file_path + f"/teams/teams.csv")

        # from playermatchstats, get stats
        gameweek_frames = []
        for i in range(self.gw_start, self.window+1):
            player_match_stats = pd.read_csv(self.file_path + f"/playermatchstats/GW{i}/playermatchstats.csv")
#            self._validate_df(player_match_stats)
            player_match_stats = self._process_df(player_match_stats)
            save_features = self.features.names
            save_features.append(self.global_id)

            gameweek_frames.append(player_match_stats[save_features])

        all_gameweeks = pd.concat(gameweek_frames, ignore_index=True)
        self.cumulative_player_stats = (
            all_gameweeks
            .groupby(self.global_id)
            .sum()
            .loc[lambda df: df[self.canonical_feature.name] >= self.threshold]
            .reset_index()
        )

        # now we have season cumulative stats it's time to add in missing stats, from vaastav repository
        vaastav = pd.read_csv(self.file_path + "/players_raw(vaastav).csv").rename(columns=self.vaastav_map)


        vaastav = self._process_df(vaastav[self.vaastav_features])


    def Canonicalise(self):
        
        canon_col = self.canonical_feature.name
        for spec in self.features.specs:
            if spec.name != canon_col:
                self.cumulative_player_stats[spec.name] = (
                    self.cumulative_player_stats[spec.name] / self.cumulative_player_stats[canon_col]
                )


class Priors: 

    def __init__():
        pass
    
class PriorCompute:
    """
    This class computes priors for f4 different heirachy levels:
                                                        - league
                                                        - position
                                                        - position x team
                                                        - individual
  
    Uses full 38GW season data to avoid any biases due to time of season
    A feature is either p90 or snapshot
    Where per 90 (p90) stats are calculated from pooled totals not averages.
    """
    
    def __init__(self):
        pass

class PriorStore:
    """
    Lookup player code, work out heirachy level, then assign data
    Data should be outputted in same form as actual game week data
    All prior data has label is_prior = 1
    """

    def __init__(self):
        pass
