"""
End-to-end smoke test for the FPL pipeline.

Wires up Features → Ingester → SeasonSequencer → build_player_window()
on a small GW range to verify the full data path produces valid tensors.
"""

import logging
import pathlib

import pandas as pd

from .feature_registry import build_features
from .features import DataSource
from .ingester import FPLSourceConfig, FixtureSourceConfig
from .sequencer import SeasonSequencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
# Resolve relative to the repo root (one level above fpl-bot/).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = _REPO_ROOT / "Data" / "FPL Data" / "2025-2026"
SEASON_ROOT = str(DATA_ROOT / "By Gameweek") + "/"

# ── Position mapping ────────────────────────────────────────────────────────
# players.csv uses full names; the sequencer and priors expect short codes.
_POSITION_SHORT = {
    "Goalkeeper": "GKP",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}


def main():
    """Run the end-to-end pipeline smoke test."""
    # 1. Load player and team metadata
    players = pd.read_csv(DATA_ROOT / "players.csv")
    players["position"] = players["position"].map(_POSITION_SHORT)
    teams = pd.read_csv(DATA_ROOT / "teams.csv")

    # 2. Build feature registry (team codes drive categorical embeddings)
    team_codes_list = teams["code"].tolist()
    features = build_features(team_codes_list)
    logger.info(
        "Features built: %d temporal, %d categorical",
        len(features.temporal_columns),
        len(features.categorical_columns),
    )

    # 3. Build id_map (player_id → player_code + team_code) for Ingester joins
    id_map = players[["player_id", "player_code", "team_code"]].copy()

    # 4. Build team_codes DataFrame for fixture right-join
    team_codes = teams[["code"]].rename(columns={"code": "team_code"})

    # 5. Build column maps from the feature registry
    fci_map = features.source_map(DataSource.FCI)
    opta_map = features.source_map(DataSource.OPTA)

    # 6. Configure data sources
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

    # 7. Player metadata for sequencer (position + team_code only)
    player_meta = players[["player_code", "player_id", "team_code", "position"]].copy()

    # 8. Construct sequencer
    seq = SeasonSequencer(
        features=features,
        season_root=SEASON_ROOT,
        fpl_config_season=fci_cfg,
        opta_config=opta_cfg,
        fixture_config=fixture_cfg,
        player_meta=player_meta,
        teams_df=teams,
        window_size=8,
    )

    # 9. Ingest a small GW range
    logger.info("Ingesting GW1-5...")
    seq.ingest_player_range(1, 5)
    logger.info("Ingestion complete. %d players tracked.", len(seq._first_gw))

    # 10. Pick a player who appeared in GW1
    gw1_players = [pid for pid, gw in seq._first_gw.items() if gw == 1]
    if not gw1_players:
        logger.error("No players found in GW1.")
        return

    test_player = gw1_players[0]
    logger.info("Building window for player: %s", test_player)

    # 11. Build a player window
    temporal, categorical = seq.build_player_window(test_player, [1, 2, 3, 4])

    logger.info("Temporal tensor shape:     %s", temporal.shape)
    logger.info("Categorical tensor shape:  %s", categorical.shape)
    logger.info("Temporal sample (row 0):   %s", temporal[0])
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
