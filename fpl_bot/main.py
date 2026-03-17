"""
End-to-end smoke test for the FPL pipeline.

Wires up Features → Ingester → SeasonSequencer → build_player_window()
on a small GW range to verify the full data path produces valid tensors.

Two seasons are configured:
    24-25: Vaastav (FPL API) + FCI (Opta) + FCI (fixtures)
    25-26: FCI (FPL API) + FCI (Opta) + FCI (fixtures)

Each season gets its own SeasonSequencer instance.
"""

import logging
import pathlib

import pandas as pd

from fpl_bot import (
    FeatureType,
    ScalingMode,
    AccumulationType,
    DataSource,
    FeatureSpec,
    Features,
    FeatureScaler,
    Ingester,
    GameweekProvider,
    FPLSourceConfig,
    FixtureSourceConfig,
    SeasonSequencer,
    PriorData,
    PriorComputer,
    build_features,
    _build_specs,
    player_team_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
# Resolve relative to the repo root (one level above fpl-bot/).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FPL_DATA = _REPO_ROOT / "Data" / "FPL Data"

DATA_ROOT_24 = _FPL_DATA / "2024-2025"
SEASON_ROOT_24 = str(DATA_ROOT_24) + "/"

DATA_ROOT_25 = _FPL_DATA / "2025-2026"
SEASON_ROOT_25 = str(DATA_ROOT_25 / "By Gameweek") + "/"

# ── Position mapping ────────────────────────────────────────────────────────
# players.csv uses full names; the sequencer and priors expect short codes.
_POSITION_SHORT = {
    "Goalkeeper": "GKP",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}


def _build_player_meta(
    players: pd.DataFrame,
    features,
) -> pd.DataFrame:
    """
    Build player metadata DataFrame with position_idx derived from registry.

    Args:
        players: Raw players DataFrame with position already mapped to short codes.
        features: Feature registry used to derive position encoding.

    Returns:
        DataFrame with player_code, player_id, team_code, position, position_idx.
    """
    meta = players[["player_code", "player_id", "team_code", "position"]].copy()

    pos_categories = features["position"].categories
    pos_to_idx = {cat: i + 1 for i, cat in enumerate(pos_categories)}
    meta["position_idx"] = meta["position"].map(pos_to_idx)

    return meta


def _build_season_24(features, id_map, team_codes):
    """
    Build configs for the 2024-2025 season.

    24-25 uses Vaastav for FPL API data (stacked merged_gw.csv),
    FCI for Opta (per-GW playermatchstats), and FCI for fixtures.

    Args:
        features: Feature registry.
        id_map: Player identity mapping DataFrame.
        team_codes: Team codes DataFrame for fixture right-join.

    Returns:
        Tuple of (vaastav_cfg, opta_cfg, fixture_cfg).
    """
    vaastav_map = features.source_map(DataSource.VAASTAV)
    opta_map = features.source_map(DataSource.OPTA)

    vaastav_cfg = FPLSourceConfig(
        provider=DataSource.VAASTAV,
        col_map=vaastav_map,
        player_id={"element": "player_id"},
        id_map=id_map,
        stacked=True,
        denotes_epl={},
        other_games=False,
        gw_col="GW",
        gw_path="vaastav/merged_gw.csv",
        gw_filename=None,
        transform=None,
    )

    opta_cfg = FPLSourceConfig(
        provider=DataSource.OPTA,
        col_map=opta_map,
        player_id={"player_id": "player_id"},
        id_map=id_map,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="playermatchstats/",
        gw_filename="playermatchstats.csv",
        transform=None,
    )

    fixture_cfg = FixtureSourceConfig(
        provider=DataSource.FIXTURE,
        col_map={},
        team_codes=team_codes,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="matches/",
    )

    return vaastav_cfg, opta_cfg, fixture_cfg


def _build_season_25(features, id_map, team_codes):
    """
    Build configs for the 2025-2026 season.

    25-26 uses FCI for everything: FPL API data, Opta, and fixtures.
    All files live under By Gameweek/GW{n}/.

    Args:
        features: Feature registry.
        id_map: Player identity mapping DataFrame.
        team_codes: Team codes DataFrame for fixture right-join.

    Returns:
        Tuple of (fci_cfg, opta_cfg, fixture_cfg).
    """
    fci_map = features.source_map(DataSource.FCI)
    opta_map = features.source_map(DataSource.OPTA)

    fci_cfg = FPLSourceConfig(
        provider=DataSource.FCI,
        col_map=fci_map,
        player_id={"id": "player_id"},
        id_map=id_map,
        stacked=False,
        denotes_epl={},
        other_games=False,
        gw_col=None,
        gw_path=None,
        gw_filename="playerstats.csv",
        transform=None,
    )

    opta_cfg = FPLSourceConfig(
        provider=DataSource.OPTA,
        col_map=opta_map,
        player_id={"player_id": "player_id"},
        id_map=id_map,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path=None,
        gw_filename="playermatchstats.csv",
        transform=None,
    )

    fixture_cfg = FixtureSourceConfig(
        provider=DataSource.FIXTURE,
        col_map={},
        team_codes=team_codes,
        stacked=False,
        denotes_epl={"tournament": "Premier League"},
        other_games=True,
        gw_col=None,
        gw_path="",
    )

    return fci_cfg, opta_cfg, fixture_cfg


def main():
    """Run the end-to-end pipeline smoke test."""
    # ── 2024-2025 season ────────────────────────────────────────────────────
    players_24 = pd.read_csv(DATA_ROOT_24 / "players" / "players.csv")
    players_24["position"] = players_24["position"].map(_POSITION_SHORT)
    teams_24 = pd.read_csv(DATA_ROOT_24 / "teams" / "teams.csv")

    team_codes_24 = teams_24["code"].tolist()
    features = build_features(team_codes_24)
    logger.info(
        "Features built: %d temporal, %d categorical",
        len(features.temporal_columns),
        len(features.categorical_columns),
    )

    id_map_24 = players_24[["player_id", "player_code", "team_code"]].copy()
    team_codes_df_24 = teams_24[["code"]].rename(columns={"code": "team_code"})
    player_meta_24 = (
        _build_player_meta(players_24, features)
        .assign(player_team_id=player_team_index)               # adds column "{player_code}_{team_code}"
        .set_index("player_team_id")                            # indexes by new composite index column
    )
    
    fpl_cfg_24, opta_cfg_24, fixture_cfg_24 = _build_season_24(
        features, id_map_24, team_codes_df_24,
    )

    seq_24 = SeasonSequencer(
        features=features,
        season_root=SEASON_ROOT_24,
        fpl_config_season=fpl_cfg_24,
        opta_config=opta_cfg_24,
        fixture_config=fixture_cfg_24,
        player_meta=player_meta_24,
        teams_df=teams_24,
        window_size=8,
    )

    logger.info("Ingesting 24-25 GW1-5...")
    seq_24.ingest_player_range(1, 5)
    logger.info("Ingestion complete. %d players tracked.", len(seq_24._first_gw))

    # Pick a player who appeared in GW1
    gw1_players = [pid for pid, gw in seq_24._first_gw.items() if gw == 1]
    if not gw1_players:
        logger.error("No players found in GW1.")
        return

    test_player = gw1_players[0]
    logger.info("Building window for player: %s", test_player)

    temporal, categorical = seq_24.build_player_window(test_player, 5)

    logger.info("Temporal tensor shape:     %s", temporal.shape)
    logger.info("Categorical tensor shape:  %s", categorical.shape)
    logger.info("Temporal sample (row 0):   %s", temporal[0])
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
