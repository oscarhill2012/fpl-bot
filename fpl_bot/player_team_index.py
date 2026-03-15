import pandas as pd 

def player_team_index(df: pd.DataFrame) -> pd.Series:
    """
    Build a composite player-team identifier for each row in the dataframe.

    Combines ``player_code`` and ``team_code`` into a single string key of the
    form ``"{player_code}_{team_code}"``. This uniquely identifies a player
    within the context of a specific team, which is useful when tracking
    players who move clubs across seasons.

    Args:
        df: Dataframe containing integer-valued ``player_code`` and
            ``team_code`` columns.

    Returns:
        String series of composite identifiers, one per row.
    """
    return df["player_code"].astype(int).astype(str) + "_" + df["team_code"].astype(int).astype(str)
