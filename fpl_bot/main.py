import logging
import pathlib

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fpl_bot import (
    DataSource,
    Features,
    FPLSourceConfig,
    FPLPointsPredictor,
    FixtureSourceConfig,
    SeasonSequencer,
    Trainer,
    build_features24,
    build_features25,
    player_team_index,
    FeatureScaler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
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
    meta["position"] = meta["position"].map(pos_to_idx)

    return (
        meta
        .assign(player_team_id=player_team_index)
        .set_index("player_team_id")
    )


# =============================================================================
# Season setup
# =============================================================================


def _setup_season_24() -> tuple[str, pd.DataFrame, pd.DataFrame, FPLSourceConfig, FPLSourceConfig, FixtureSourceConfig]:
    """
    Load metadata and build configs for the 2024-2025 season.

    24-25 data is split across three provider folders, each with its own
    base file for player identity mapping:
        FPL_API/  -- Vaastav (stacked merged_gw.csv), base: player_idlist.csv
        OPTA/     -- per-GW playermatchstats, base: players.csv
        Fixtures/ -- per-GW matches, base: teams.csv

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
        transform={"price": lambda x: x / 10},
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

    return season_root, player_meta, team_codes_df, fpl_cfg, opta_cfg, fixture_cfg


def _setup_season_25() -> tuple[str, pd.DataFrame, pd.DataFrame, FPLSourceConfig, FPLSourceConfig, FixtureSourceConfig]:
    """
    Load metadata and build configs for the 2025-2026 season.

    25-26 uses a single provider (FCI) with all files under GW{n}/:
        player_gameweek_stats.csv  -- FPL API data
        playermatchstats.csv       -- OPTA data
        fixtures.csv               -- fixture data

    Player and team metadata live in the season root folder.

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

    return season_root, player_meta, team_codes_df, fpl_cfg, opta_cfg, fixture_cfg

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

# categorical features for position, 0 is left for padding
_PLAYER_POS_ID = {
    "GKP": 1,
    "DEF": 2,
    "MID": 3,
    "FWD": 4,
}

def main():
    
    # ─── 1. Set up Seasons ───
    # ─── 1(a). 25/26 ───

    season_root, players, teams, fpl_cfg, opta_cfg, fixture_cfg = _setup_season_25()
    features25 = build_features25(list(teams["team_code"]))

    player_meta = _build_player_meta(players, features25)
    
    # ─── 1(b). 24/25 ───
    
    # we inherit player_meta from 25/26 as we only need those players priors anyway

    prior_season_root, defunc_players, prior_teams, prior_fpl_cfg, prior_opta_cfg, prior_fixture_cfg = _setup_season_24()
    features24 = build_features24(list(prior_teams["team_code"]))

    features24._spec_by_name["position"].categories = [1, 2, 3, 4]

    # ─── 2. Get Priors ───

    prior_seq = SeasonSequencer(
        features=features24,
        season_root=prior_season_root,
        fpl_config_season=prior_fpl_cfg,
        opta_config=prior_opta_cfg,
        fixture_config=prior_fixture_cfg,
        player_meta=player_meta,
    )

    prior_seq.ingest_range(1, 38)
    priors = prior_seq.get_prior

    # ─── 3. Set-up Current Season ───

    seq = SeasonSequencer(
        features=features25,
        season_root=season_root,
        fpl_config_season=fpl_cfg,
        opta_config=opta_cfg,
        fixture_config=fixture_cfg,
        player_meta=player_meta,
        prior_data=priors
    )

    seq.ingest_range(1, 30)

    # ─── 4. Create train/val datasets with temporal split ───
    # GW 2-24 for training, GW 25-29 for validation
    # (GW 1 has no history, GW 30 is the last ingested target)

    train_ds = seq.dataset(gw_start=2, gw_end=23) 
    val_ds = seq.dataset(gw_start=24, gw_end=29)
    

    print(f"Train samples: {len(train_ds)}")  # n_players * 23
    print(f"Val samples:   {len(val_ds)}")    # n_players * 5

    # ─── 5. Create DataLoaders ───

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)
 
    # ─── 6. Fit the scaler on training data ───
    # Collect all training x_numeric and position IDs for fitting.
    # Position IDs enable position-aware scaling for GK-specific features.

    all_x_numeric = []
    all_position_ids = []
    for batch in train_loader:
        all_x_numeric.append(batch["x_numeric"])
        # position is first categorical, constant across timesteps
        all_position_ids.append(batch["x_categorical"][:, 0, 0])

    # Stack into [N_total, T, F] — this IS the [P, G, F] convention
    all_x_numeric = torch.cat(all_x_numeric, dim=0)
    position_ids = torch.cat(all_position_ids, dim=0)

    scaled_features = features25.filtered_numeric
    scaler = FeatureScaler(scaled_features)
    scaler.train_scale(all_x_numeric, position_ids=position_ids)
    # scaler is now fitted — test_scale() will apply these params per batch

    # ─── 7. Build model from feature registry ───

    model = FPLPointsPredictor.from_features(features25)

    # ─── 8. Train ───

    trainer = Trainer(
        model=model,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        grad_clip=1.0,
    )

    history = trainer.fit(epochs=100, patience=15)

if __name__ == "__main__":
    main()
