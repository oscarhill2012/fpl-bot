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
    Features,
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
    meta["position_idx"] = meta["position"].map(pos_to_idx)

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

# categorical features for postion, 0 is left for padding
_PLAYER_POS_ID = {
    "GK": 1,
    "DEF": 2,
    "MID": 3,
    "FWD": 4,
}

def main():
    
    # ─── 1. Set up Seasons ───
    # ─── 1(a). 25/26 ───

    season_root, player_meta, teams, fpl_cfg, opta_cfg, fixture_cfg = _setup_season_25()
    features25 = build_features25(list(teams["team_code"]))

    player_meta["position"] = player_meta["position"].map(_PLAYER_POS_ID)
    features25._spec_by_name["position"].categories = [1, 2, 3, 4]
    
    # ─── 1(b). 24/25 ───
    
    # we inherit player_meta from 25/26 as we only need those players priors anyway

    prior_season_root, defunc_player_meta, prior_teams, prior_fpl_cfg, prior_opta_cfg, prior_fixture_cfg = _setup_season_24()
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

    seq.ingest_range(1, 16)

    # ─── 4. Create train/val datasets with temporal split ───
    # GW 2-24 for training, GW 25-29 for validation
    # (GW 1 has no history, GW 30 is the last ingested target)

    train_ds = seq.dataset(gw_start=2, gw_end=11)
    val_ds   = seq.dataset(gw_start=12, gw_end=13)

    print(f"Train samples: {len(train_ds)}")  # n_players * 23
    print(f"Val samples:   {len(val_ds)}")    # n_players * 5


    # ─── 5. Create DataLoaders ───

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ─── Smoke test: inspect pipeline output ───
    batch = next(iter(train_loader))
    x_cont = batch["x_continuous"]
    x_cat = batch["x_categorical"]
    x_fix = batch["x_future_fixtures"]
    y = batch["y"]

    print(f"\n{'=' * 50}")
    print("Sample batch from train_loader:")
    print(f"  x_continuous:      {x_cont.shape}  dtype={x_cont.dtype}")
    print(f"  x_categorical:     {x_cat.shape}  dtype={x_cat.dtype}")
    print(f"  x_future_fixtures: {x_fix.shape}  dtype={x_fix.dtype}")
    print(f"  y:                 {y.shape}  dtype={y.dtype}")
    print()
    print(f"  x_continuous NaN:  {torch.isnan(x_cont).sum().item()}")
    print(f"  x_continuous Inf:  {torch.isinf(x_cont).sum().item()}")
    print(f"  x_continuous min:  {x_cont.min().item():.4f}")
    print(f"  x_continuous max:  {x_cont.max().item():.4f}")
    print(f"  x_continuous mean: {x_cont.mean().item():.4f}")
    print(f"  x_continuous std:  {x_cont.std().item():.4f}")
    print(f"{'=' * 50}\n")

    """    # ─── 6. Fit the scaler on training data ───
    # Collect all training x_continuous into one tensor for fitting.
    # This is a one-time operation before training starts.

    all_x_continuous = []
    for batch in train_loader:
        all_x_continuous.append(batch["x_continuous"])

    # Stack into [N_total, T, F] — this IS the [P, G, F] convention
    all_x_continuous = torch.cat(all_x_continuous, dim=0)

    scaler = FeatureScaler(features)
    scaled_train, features_dict = scaler.train_scale(all_x_continuous)
    # scaler is now fitted — scaling parameters stored in each FeatureSpec


    # ─── 5. Training loop ───

    model = ...       # your LSTM + MLP model
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            # Unpack the batch — each value is a tensor with batch dim 0
            x_cont = batch["x_continuous"]        # [B, T, F_cont]
            x_cat  = batch["x_categorical"]       # [B, T, C]
            x_fix  = batch["x_future_fixtures"]   # [B, K, fix_features]
            y      = batch["y"]                   # [B, target_window_size]

            # Scale continuous features using the fitted scaler
            # test_scale uses parameters from train_scale — no data leakage
            x_scaled = scaler.test_scale(x_cont)  # [B, T, F_cont]

            # Forward pass — model handles embedding internally
            pred = model(x_scaled, x_cat, x_fix)  # [B, 1]
            loss = criterion(pred, y)

            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # ─── Validation ───
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["x_continuous"]
                x_cat  = batch["x_categorical"]
                x_fix  = batch["x_future_fixtures"]
                y      = batch["y"]

                x_scaled = scaler.test_scale(x_cont)
                pred = model(x_scaled, x_cat, x_fix)
                val_loss += criterion(pred, y).item()

        print(f"Epoch {epoch}: val_loss = {val_loss / len(val_loader):.4f}")
    """

if __name__ == "__main__":
    main()
