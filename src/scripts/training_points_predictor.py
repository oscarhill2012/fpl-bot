import logging
import math
import pathlib
import random 

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fpl_bot import (
    DataSource,
    Features,
    FPLSourceConfig,
    FixtureSourceConfig,
    SeasonSequencer,
    build_features24,
    build_features25,
    player_team_index,
    FeatureScaler,
    Trainer,
    FPLPointsPredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# -- SEED ---------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -- Paths ---------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "Data"

DATA_ROOT_24 = _DATA_ROOT / "2024-2025"
DATA_ROOT_25 = _DATA_ROOT / "2025-2026"

# -- Position mapping ---------------------------------------------------------
# Player metadata uses full names; the sequencer and priors expect short codes.
_POSITION_TO_NUM = {
    "Goalkeeper": 1,
    "Defender": 2,
    "Midfielder": 3,
    "Forward": 4,
}

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

    return season_root, team_codes_df, fpl_cfg, opta_cfg, fixture_cfg


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
    player_meta["position"] = player_meta["position"].map(_POSITION_TO_NUM)

    player_meta["player_team_id"] = player_team_index(player_meta)   # shared util
    player_meta = player_meta.set_index("player_team_id")

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

    season_root, player_meta, teams, fpl_cfg, opta_cfg, fixture_cfg = _setup_season_25()
    features25 = build_features25(list(teams["team_code"]))
    
    # ─── 1(b). 24/25 ───
    
    # we inherit player_meta from 25/26 as we only need those players priors anyway

    prior_season_root, prior_teams, prior_fpl_cfg, prior_opta_cfg, prior_fixture_cfg = _setup_season_24()
    features24 = build_features24(list(prior_teams["team_code"]))

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
        prior_data=priors, 
        window_size=8,
        predict_window_size=1,
    )

    seq.ingest_range(1, 30)

    # ─── 4. Create train/val datasets with temporal split ───
    # via random player split

    all_players = list(seq._player_meta.keys())
    rng = random.Random(SEED)
    rng.shuffle(all_players)
    
    split = int(0.8 * len(all_players))
    train_players = all_players[:split]
    val_players   = all_players[split:]

    train_ds = seq.dataset(gw_start=2, gw_end=25, player_codes=train_players) 
    val_ds = seq.dataset(gw_start=2, gw_end=25, player_codes=val_players)
    
    print(f"Train samples: {len(train_ds)}") 
    print(f"Val samples:   {len(val_ds)}")    

    # ─── 5. Create DataLoaders ───

    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── 5(a). Cache raw samples ───
    # Build all samples once; avoids repeated build_player_window calls.
    train_ds.cache()
    val_ds.cache()

    # ─── 5(b). Fit scaler on training data and scale both splits ───
    all_x_numeric = train_ds.stacked_numeric()
    position_ids = torch.tensor([
        player_meta.loc[pid, "position"]
        for pid, _ in train_ds.sample_index
    ])

    scaled_features = features25.filtered_numeric
    scaler = FeatureScaler(scaled_features, device=device)
    scaled_train, _ = scaler.train_scale(
        all_x_numeric, position_ids=position_ids,
    )

    # ─── 5(c). Scale cached samples ───
    # Training data uses train_scale result directly; validation via test_scale.
    train_ds.apply_scaled(scaled_train.cpu())
    val_ds.apply_scaler(scaler)

    # ─── 5(d). Scale targets ───
    # Use the same ROBUST params fitted for the "points" input feature.
    points_median, points_iqr = scaler.get_params("points")
    train_ds.scale_targets(points_median, points_iqr)
    val_ds.scale_targets(points_median, points_iqr)

    # ─── 5(e). Scale fixture features ───
    # ELO features in x_future_fixtures use the same params as the LSTM input.
    train_ds.scale_fixtures(scaler)
    val_ds.scale_fixtures(scaler)

    # ─── 5(f). Create DataLoaders ───
    # Created AFTER scaling so persistent workers see the cached data.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,

    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,

    )

    # ─── 7(a). Build model ───
    # All model and training parameters live here for easy grid search.

    # Model architecture
    lstm_hidden_dim = 128
    lstm_layers = 2
    mlp_hidden_dim = 64
    dropout = 0.1
    n_fixture_features = 5

    # Training
    learning_rate = 5e-4
    weight_decay = 1e-5
    grad_clip = 1.0
    epochs = 100
    patience = 8

    # ─── 7(b). Build model ───

    model = FPLPointsPredictor.from_features(
        features25,
        n_fixture_features=n_fixture_features,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
    )

    # (optional but recommended debug)
    print(f"Using device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")


    # ─── 8. Initialise Trainer ───

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        device=device,
        target_iqr=points_iqr,
        target_median=points_median,
    )

    # ─── 9. Train ───
    history = trainer.fit(
        epochs=epochs,
        patience=patience,
        run_version="testing",
    )

    # ─── 10. (Optional) Save final model explicitly ───

    # torch.save(model.state_dict(), "final_model_manual.pt")


if __name__ == "__main__":
    main()
