"""
End-to-end smoke test for the FPL pipeline.

Wires up Features -> Ingester -> SeasonSequencer -> build_player_window()
on a small GW range to verify the full data path produces valid tensors.

Two seasons are configured:
    24-25: Vaastav (FPL API) + OPTA + Fixtures
    25-26: FCI (FPL API) + OPTA + Fixtures

Each season gets its own SeasonSequencer instance.
"""

import logging
import pathlib

import pandas as pd

from fpl_bot import (
    DataSource,
    Features,
    FPLSourceConfig,
    FixtureSourceConfig,
    SeasonSequencer,
    build_features,
    player_team_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -- Paths -------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATA_ROOT = _PROJECT_ROOT / "Data"

DATA_ROOT_24 = _DATA_ROOT / "2024-2025"
DATA_ROOT_25 = _DATA_ROOT / "2025-2026"

# -- Position mapping ---------------------------------------------------------
# Player metadata uses full names; the sequencer and priors expect short codes.
_POSITION_SHORT = {
    "Goalkeeper": "GKP",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}


def _build_player_meta(
    players: pd.DataFrame,
    features: Features,
) -> pd.DataFrame:
    """
    Build player metadata DataFrame with position_idx derived from registry.

    Args:
        players: Raw players DataFrame with position already mapped to short codes.
        features: Feature registry used to derive position encoding.

    Returns:
        DataFrame indexed by player_team_id with player_code, player_id,
        team_code, position, and position_idx columns.
    """
    meta = players[["player_code", "player_id", "team_code", "position"]].copy()

    pos_categories = features["position"].categories
    pos_to_idx = {cat: i + 1 for i, cat in enumerate(pos_categories)}
    meta["position_idx"] = meta["position"].map(pos_to_idx)

    return (
        meta
        .assign(player_team_id=player_team_index)
        .set_index("player_team_id")
    )


# =============================================================================
# Season setup
# =============================================================================


def _setup_season_24(
    features: Features,
) -> tuple[str, pd.DataFrame, pd.DataFrame, FPLSourceConfig, FPLSourceConfig, FixtureSourceConfig]:
    """
    Load metadata and build configs for the 2024-2025 season.

    24-25 data is split across three provider folders, each with its own
    base file for player identity mapping:
        FPL_API/  -- Vaastav (stacked merged_gw.csv), base: player_idlist.csv
        OPTA/     -- per-GW playermatchstats, base: players.csv
        Fixtures/ -- per-GW matches, base: teams.csv

    Args:
        features: Feature registry built from this season's team codes.

    Returns:
        Tuple of (season_root, player_meta, teams_df,
        fpl_cfg, opta_cfg, fixture_cfg).
    """
    root = DATA_ROOT_24
    season_root = str(root) + "/"

    # -- Provider base files --------------------------------------------------
    vaastav_base = pd.read_csv(root / "FPL_API" / "player_idlist.csv")
    opta_base = pd.read_csv(root / "OPTA" / "players.csv")
    teams = pd.read_csv(root / "Fixtures" / "teams.csv")

    # -- ID maps --------------------------------------------------------------
    # Each provider gets its own id_map as a guard against mismatched IDs.
    # Vaastav uses "id" as its player identifier; join with OPTA base to
    # resolve player_code and team_code.
    vaastav_id_map = (
        vaastav_base
        .rename(columns={"id": "player_id"})
        .merge(
            opta_base[["player_id", "player_code", "team_code"]],
            on="player_id",
        )
        [["player_id", "player_code", "team_code"]]
    )
    opta_id_map = opta_base[["player_id", "player_code", "team_code"]].copy()

    team_codes_df = teams[["code"]].rename(columns={"code": "team_code"})

    # -- Player metadata ------------------------------------------------------
    # OPTA base has position; map to short codes for the sequencer.
    player_meta = opta_base[
        ["player_code", "player_id", "team_code", "position"]
    ].copy()
    player_meta["position"] = player_meta["position"].map(_POSITION_SHORT)

    # -- Source configs -------------------------------------------------------
    fpl_cfg = FPLSourceConfig(
        provider=DataSource.VAASTAV,
        player_id={"element": "player_id"},
        id_map=vaastav_id_map,
        stacked=True,
        denotes_epl={},
        other_games=False,
        gw_col="GW",
        gw_path="FPL_API/merged_gw.csv",
        gw_filename=None,
        transform=None,
    )

    opta_cfg = FPLSourceConfig(
        provider=DataSource.OPTA,
        player_id={"player_id": "player_id"},
        id_map=opta_id_map,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="OPTA/",
        gw_filename="playermatchstats.csv",
        transform=None,
    )

    fixture_cfg = FixtureSourceConfig(
        provider=DataSource.FIXTURE,
        team_codes=team_codes_df,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="Fixtures/",
        gw_filename="matches.csv",
    )

    return season_root, player_meta, teams, fpl_cfg, opta_cfg, fixture_cfg


def _setup_season_25(
    features: Features,
) -> tuple[str, pd.DataFrame, pd.DataFrame, FPLSourceConfig, FPLSourceConfig, FixtureSourceConfig]:
    """
    Load metadata and build configs for the 2025-2026 season.

    25-26 uses a single provider (FCI) with all files under GW{n}/:
        player_gameweek_stats.csv  -- FPL API data
        playermatchstats.csv       -- OPTA data
        fixtures.csv               -- fixture data

    Player and team metadata live in the season root folder.

    Args:
        features: Feature registry built from this season's team codes.

    Returns:
        Tuple of (season_root, player_meta, teams_df,
        fpl_cfg, opta_cfg, fixture_cfg).
    """
    root = DATA_ROOT_25
    season_root = str(root) + "/"

    # -- Metadata -------------------------------------------------------------
    players = pd.read_csv(root / "players.csv")
    teams = pd.read_csv(root / "teams.csv")

    # -- ID maps --------------------------------------------------------------
    # Both FPL API and OPTA share the same players.csv for this season.
    # Separate copies as a guard against accidental mutation.
    fci_id_map = players[["player_id", "player_code", "team_code"]].copy()
    opta_id_map = players[["player_id", "player_code", "team_code"]].copy()

    team_codes_df = teams[["code"]].rename(columns={"code": "team_code"})

    # -- Player metadata ------------------------------------------------------
    player_meta = players[
        ["player_code", "player_id", "team_code", "position"]
    ].copy()
    player_meta["position"] = player_meta["position"].map(_POSITION_SHORT)

    # -- Source configs -------------------------------------------------------
    fpl_cfg = FPLSourceConfig(
        provider=DataSource.FCI,
        player_id={"id": "player_id"},
        id_map=fci_id_map,
        stacked=False,
        denotes_epl={},
        other_games=False,
        gw_col=None,
        gw_path="",
        gw_filename="player_gameweek_stats.csv",
        transform=None,
    )

    opta_cfg = FPLSourceConfig(
        provider=DataSource.OPTA,
        player_id={"player_id": "player_id"},
        id_map=opta_id_map,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="",
        gw_filename="playermatchstats.csv",
        transform=None,
    )

    fixture_cfg = FixtureSourceConfig(
        provider=DataSource.FIXTURE,
        team_codes=team_codes_df,
        stacked=False,
        denotes_epl={"match_id": "prem"},
        other_games=True,
        gw_col=None,
        gw_path="",
        gw_filename="fixtures.csv",
    )

    return season_root, player_meta, teams, fpl_cfg, opta_cfg, fixture_cfg


# =============================================================================
# Season runner
# =============================================================================


def _run_season(
    label: str,
    season_root: str,
    features: Features,
    player_meta: pd.DataFrame,
    teams: pd.DataFrame,
    fpl_cfg: FPLSourceConfig,
    opta_cfg: FPLSourceConfig,
    fixture_cfg: FixtureSourceConfig,
    gw_start: int,
    gw_end: int,
) -> None:
    """
    Build a SeasonSequencer, ingest a GW range, and build one player window.

    Args:
        label: Human-readable season label for logging (e.g. "24-25").
        season_root: Path to the season data directory.
        features: Feature registry for this season.
        player_meta: Raw player metadata with position mapped to short codes.
        teams: Teams DataFrame with code, name, elo, etc.
        fpl_cfg: FPL API source config.
        opta_cfg: OPTA source config.
        fixture_cfg: Fixture source config.
        gw_start: First gameweek to ingest (inclusive).
        gw_end: Last gameweek to ingest (inclusive).
    """
    meta = _build_player_meta(player_meta, features)

    seq = SeasonSequencer(
        features=features,
        season_root=season_root,
        fpl_config_season=fpl_cfg,
        opta_config=opta_cfg,
        fixture_config=fixture_cfg,
        player_meta=meta,
        teams_df=teams,
        window_size=8,
    )

    logger.info("Ingesting %s GW%d-%d...", label, gw_start, gw_end)
    seq.ingest_player_range(gw_start, gw_end)
    logger.info("Ingestion complete. %d players tracked.", len(seq._first_gw))

    # Pick a player who appeared in the first gameweek.
    gw1_players = [pid for pid, gw in seq._first_gw.items() if gw == gw_start]
    if not gw1_players:
        logger.error("No players found in GW%d (%s).", gw_start, label)
        return

    test_player = gw1_players[0]
    logger.info("Building window for player: %s", test_player)

    temporal, categorical = seq.build_player_window(test_player, gw_end)

    logger.info("Temporal tensor shape:     %s", temporal.shape)
    logger.info("Categorical tensor shape:  %s", categorical.shape)
    logger.info("Temporal sample (row 0):   %s", temporal[0])
    logger.info("%s smoke test passed.", label)


# =============================================================================
# Private helpers
# =============================================================================


def _resolve_teams_path(data_root: pathlib.Path) -> pathlib.Path:
    """Locate teams.csv by checking known season folder layouts."""
    candidates = [
        data_root / "teams.csv",
        data_root / "Fixtures" / "teams.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"teams.csv not found in {data_root}. "
        f"Checked: {[str(c) for c in candidates]}"
    )


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the end-to-end pipeline smoke test for each season."""
    seasons = [
        ("24-25", DATA_ROOT_24, _setup_season_24),
        ("25-26", DATA_ROOT_25, _setup_season_25),
    ]

    for label, data_root, setup_fn in seasons:
        # Load teams first to build Features (needs team codes for embeddings).
        teams_path = _resolve_teams_path(data_root)
        teams = pd.read_csv(teams_path)
        team_codes = teams["code"].tolist()
        features = build_features(team_codes)

        logger.info(
            "%s — Features built: %d temporal, %d categorical",
            label,
            len(features.temporal_columns),
            len(features.categorical_columns),
        )

        season_root, player_meta, teams_df, fpl_cfg, opta_cfg, fixture_cfg = (
            setup_fn(features)
        )

        _run_season(
            label, season_root, features, player_meta, teams_df,
            fpl_cfg, opta_cfg, fixture_cfg,
            gw_start=1, gw_end=1,
        )

    # -- Feature sanity check -------------------------------------------------
    logger.info("--- Feature Sanity Check (last season) ---")
    logger.info("Temporal features (%d columns):", len(features.temporal_columns))
    for i, name in enumerate(features.temporal_columns):
        logger.info("  [%d] %s", i, name)

    logger.info(
        "Categorical features (%d columns):", len(features.categorical_columns),
    )
    for i, name in enumerate(features.categorical_columns):
        logger.info("  [%d] %s", i, name)


if __name__ == "__main__":
    main()
