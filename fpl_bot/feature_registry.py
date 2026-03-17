"""
Feature Registry — single source of truth for all model features.

Every FeatureSpec defined here flows through the entire pipeline:
    Ingester  → reads source_columns + accumulation to know what to load / accumulate
    Priors    → reads per_90_columns / snapshot_columns to compute hierarchical averages
    Sequencer → reads temporal_columns / categorical_columns for tensor layout
    Scaler    → reads scaling_mode + feature_type to pick the right transform
    Model     → reads categorical specs for nn.Embedding dimensions

Ordering matters: the position of each spec in FEATURE_SPECS defines
its column index in the output tensors. Temporal specs come first
(they form x_temporal), categoricals last (they form x_categorical).

Naming convention:
    - Features that the ingester accumulates and divides by minutes
      are named  *_per_90  (e.g. "goals_per_90").
    - Their derived cumulative column (via FeatureSpec.cum_col) strips
      "_per_90" and prepends "cum_":  "goals_per_90" → "cum_goals".
    - RAW_CUMULATIVE features just prepend "cum_":  "minutes" → "cum_minutes".
"""

from fpl_bot.features import (
    FeatureSpec,
    Features,
    FeatureType,
    ScalingMode,
    AccumulationType,
    DataSource,
)



def _build_specs(team_codes: list[int]) -> list[FeatureSpec]:
    """
    Build the canonical feature spec list.

    Args:
        team_codes: Integer team codes from teams.csv (e.g. [1, 2, … 20]).
            Used as the category list for team_code and oppo_team_code
            embeddings. Index 0 is reserved as padding_idx.

    Returns:
        Ordered list of FeatureSpec objects defining tensor column positions.
    """
    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE SPECS — ordered list that defines tensor column positions
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Layout:
    #   §1  Sequencer meta-features   (is_prior, data_age)
    #   §2  FPL raw-cumulative        (minutes, starts)
    #   §3  FPL snapshots             (price, points, transfers)
    #   §4  FPL per-90 rates          (goals, assists, xG, ICT, …)
    #   §5  Opta per-90 rates         (shots, passes, tackles, GK, …)
    #   §6  Fixture features          (team_elo, oppo_team_elo, is_home)
    #   §7  Categorical embeddings    (position, team_code, oppo_team_code)
    # ═══════════════════════════════════════════════════════════════════════

    return [

    # ── §1  Sequencer meta-features ───────────────────────────────────────
    #
    # These are NOT produced by the ingester.  The sequencer stamps them
    # onto every row (real or prior) at sequence-build time.
    #   is_prior  — 1.0 for synthetic prior rows, 0.0 for real data
    #   data_age  — integer countdown (T, T-1, … 1) telling the LSTM
    #               how far back each timestep is from the prediction target
    #
    # Both are left unscaled (IDENTITY) because they carry discrete
    # structural meaning the model should see raw.

    FeatureSpec(
        name="is_prior",
        feature_type=FeatureType.BINARY,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.SEQUENCER: "is_prior"},
    ),
    FeatureSpec(
        name="data_age",
        feature_type=FeatureType.ORDINAL,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.SEQUENCER: "data_age"},
    ),

    # ── §2  FPL raw-cumulative features ───────────────────────────────────
    #
    # These are accumulated across GWs but NOT divided by minutes.
    #   minutes  — total minutes played (denominator for all per-90 rates)
    #   starts   — number of times in the starting XI
    #
    # The ingester sums these into cum_minutes / cum_starts in its
    # cumulative_df, then outputs the per-GW value in player_gw_stats.
    #
    # minutes is the most important "did this player actually play?" signal.
    # It has a heavy spike at 0 and 90, with a spread in between, so
    # ROBUST scaling handles the non-Gaussian shape well.
    #
    # starts is binary per-GW (0 or 1), but after carry-forward /
    # accumulation context it behaves like a count.  IDENTITY is fine
    # because the model mostly uses it as "started or not" per timestep.

    FeatureSpec(
        name="minutes",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.RAW_CUMULATIVE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "minutes", DataSource.FCI: "minutes", DataSource.OPTA: "minutes_played"},
    ),
    FeatureSpec(
        name="starts",

        feature_type=FeatureType.BINARY,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "starts", DataSource.FCI: "starts"},
    ),

    # ── §3  FPL snapshot features ─────────────────────────────────────────
    #
    # These are NOT accumulated — the ingester passes through the per-GW
    # value directly.  For DGWs the ingester takes the last snapshot
    # (most recent kickoff).
    #
    # price        — player cost in £m (vaastav stores tenths, transform
    #                divides by 10; FCI now_cost is already tenths but
    #                same transform applies).  Roughly Gaussian across
    #                the squad → LINEAR scaling.
    #
    # points       — FPL points scored that GW.  Can be negative (own
    #                goals, yellow cards on 0-minute subs).  Approximately
    #                Gaussian with a long right tail → LINEAR scaling.
    #
    # transfers_in / transfers_out — raw transfer counts per GW.
    #                Heavily right-skewed (most players < 1k, Haaland
    #                gets 500k+).  LOG scaling compresses the tail.

    FeatureSpec(
        name="price",

        feature_type=FeatureType.GAUSSIAN,
        scaling_mode=ScalingMode.LINEAR,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "value", DataSource.FCI: "now_cost"},
    ),
    FeatureSpec(
        name="points",

        feature_type=FeatureType.GAUSSIAN,
        scaling_mode=ScalingMode.LINEAR,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "total_points", DataSource.FCI: "event_points"},
    ),
    FeatureSpec(
        name="transfers_in",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "transfers_in", DataSource.FCI: "transfers_in"},
    ),
    FeatureSpec(
        name="transfers_out",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.VAASTAV: "transfers_out", DataSource.FCI: "transfers_out"},
    ),

    # ── §4  FPL per-90 rates ──────────────────────────────────────────────
    #
    # The ingester accumulates raw counts (cum_goals, cum_assists, …)
    # and divides by cum_minutes/90 to produce these rates.  Values are
    # carried forward for 0-minute GWs so the model always sees the
    # player's latest known production rate.
    #
    # Most rates are non-negative and right-skewed — a few elite
    # players dominate.  LOG_ROBUST (log1p then median/IQR) is ideal:
    #   • log1p compresses the tail without breaking on zeros
    #   • median/IQR is robust to the remaining outliers
    #
    # goals_conceded_per_90 and expected_goals_conceded_per_90 are also
    # non-negative (you can't concede negative goals) → same treatment.
    #
    # Shared between vaastav (24-25) and FCI (25-26) — both provide
    # exactly the same FPL API columns with minor name differences
    # handled by source_columns.

    FeatureSpec(
        name="goals_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "goals_scored", DataSource.FCI: "goals_scored"},
    ),
    FeatureSpec(
        name="assists_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "assists", DataSource.FCI: "assists"},
    ),
    FeatureSpec(
        name="clean_sheets_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "clean_sheets", DataSource.FCI: "clean_sheets"},
    ),
    FeatureSpec(
        name="goals_conceded_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "goals_conceded", DataSource.FCI: "goals_conceded"},
    ),
    FeatureSpec(
        name="yellow_cards_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "yellow_cards", DataSource.FCI: "yellow_cards"},
    ),
    FeatureSpec(
        name="saves_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "saves", DataSource.FCI: "saves"},
    ),
    FeatureSpec(
        name="bonus_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "bonus", DataSource.FCI: "bonus"},
    ),
    FeatureSpec(
        name="bps_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "bps", DataSource.FCI: "bps"},
    ),
    FeatureSpec(
        name="influence_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "influence", DataSource.FCI: "influence"},
    ),
    FeatureSpec(
        name="creativity_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "creativity", DataSource.FCI: "creativity"},
    ),
    FeatureSpec(
        name="threat_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "threat", DataSource.FCI: "threat"},
    ),
    FeatureSpec(
        name="ict_index_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "ict_index", DataSource.FCI: "ict_index"},
    ),
    FeatureSpec(
        name="expected_goals_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "expected_goals", DataSource.FCI: "expected_goals"},
    ),
    FeatureSpec(
        name="expected_assists_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.VAASTAV: "expected_assists", DataSource.FCI: "expected_assists"},
    ),
    FeatureSpec(
        name="expected_goal_involvements_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={
            DataSource.VAASTAV: "expected_goal_involvements",
            DataSource.FCI: "expected_goal_involvements",
        },
    ),
    FeatureSpec(
        name="expected_goals_conceded_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={
            DataSource.VAASTAV: "expected_goals_conceded",
            DataSource.FCI: "expected_goals_conceded",
        },
    ),

    # ── §5  Opta per-90 rates ─────────────────────────────────────────────
    #
    # Detailed match-level stats from Opta via FCI's playermatchstats.
    # Only available for 25-26 (FCI) — vaastav 24-25 has no Opta data,
    # so source_columns only has an "opta" key.  For 24-25 seasons these
    # columns are filled with 0.0 defaults by the ingester's right-join.
    #
    # The ingester accumulates raw counts into cum_shots, cum_touches, etc.
    # and divides by cum_minutes/90, exactly as for FPL per-90 rates.
    #
    # Scaling rationale:
    #   • Most Opta counts are non-negative and right-skewed → LOG_ROBUST
    #   • goals_prevented_per_90 can be negative (GK conceded more than
    #     expected) → ROBUST (no log, which would break on negatives)
    #   • BPS-like composites (influence, etc.) can have large range
    #     → ROBUST handles outliers without needing log

    # -- Attack --
    FeatureSpec(
        name="shots_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "total_shots"},
    ),
    FeatureSpec(
        name="xgot_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "xgot"},
    ),
    FeatureSpec(
        name="shots_on_target_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "shots_on_target"},
    ),
    FeatureSpec(
        name="big_chances_missed_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "big_chances_missed"},
    ),
    FeatureSpec(
        name="chances_created_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "chances_created"},
    ),
    FeatureSpec(
        name="touches_opposition_box_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "touches_opposition_box"},
    ),

    # -- Passing --
    FeatureSpec(
        name="accurate_passes_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "accurate_passes"},
    ),
    FeatureSpec(
        name="final_third_passes_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "final_third_passes"},
    ),
    FeatureSpec(
        name="accurate_crosses_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "accurate_crosses"},
    ),
    FeatureSpec(
        name="accurate_long_balls_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "accurate_long_balls"},
    ),

    # -- Possession --
    FeatureSpec(
        name="touches_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "touches"},
    ),
    FeatureSpec(
        name="successful_dribbles_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "successful_dribbles"},
    ),

    # -- Defence --
    FeatureSpec(
        name="tackles_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "tackles"},
    ),
    FeatureSpec(
        name="interceptions_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "interceptions"},
    ),
    FeatureSpec(
        name="recoveries_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "recoveries"},
    ),
    FeatureSpec(
        name="blocks_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "blocks"},
    ),
    FeatureSpec(
        name="clearances_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "clearances"},
    ),
    FeatureSpec(
        name="headed_clearances_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "headed_clearances"},
    ),
    FeatureSpec(
        name="dribbled_past_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "dribbled_past"},
    ),

    # -- Duels --
    FeatureSpec(
        name="duels_won_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "duels_won"},
    ),
    FeatureSpec(
        name="duels_lost_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "duels_lost"},
    ),
    FeatureSpec(
        name="ground_duels_won_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "ground_duels_won"},
    ),
    FeatureSpec(
        name="aerial_duels_won_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "aerial_duels_won"},
    ),

    # -- Discipline / Other --
    FeatureSpec(
        name="was_fouled_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "was_fouled"},
    ),
    FeatureSpec(
        name="fouls_committed_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "fouls_committed"},
    ),

    # -- Goalkeeper --
    FeatureSpec(
        name="xgot_faced_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "xgot_faced"},
    ),
    FeatureSpec(
        name="goals_prevented_per_90",

        feature_type=FeatureType.GAUSSIAN,
        scaling_mode=ScalingMode.ROBUST,          # can be negative → no log
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "goals_prevented"},
    ),
    FeatureSpec(
        name="sweeper_actions_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "sweeper_actions"},
    ),
    FeatureSpec(
        name="high_claim_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "high_claim"},
    ),
    FeatureSpec(
        name="gk_accurate_passes_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "gk_accurate_passes"},
    ),
    FeatureSpec(
        name="gk_accurate_long_balls_per_90",

        feature_type=FeatureType.SKEWED_POSITIVE,
        scaling_mode=ScalingMode.LOG_ROBUST,
        accumulation=AccumulationType.PER_90,
        temporal=True,
        source_columns={DataSource.OPTA: "gk_accurate_long_balls"},
    ),

    # ── §6  Fixture features ───────────────────────────────────────────────
    #
    # Match-level context from the fixtures data provider.
    # These describe the fixture, not the player's performance.
    #
    # team_elo / oppo_team_elo — Elo ratings for the player's team and
    #   their opponent at the time of the match.  Continuous floats
    #   (~1700–2000 range), roughly Gaussian across teams → LINEAR scaling.
    #   Snapshots per GW (not accumulated) — the model sees the sequence
    #   of fixture difficulty over time.
    #   Blank gameweeks (no fixture) are set to 0.0 in the data.  Real Elo
    #   is always >1000, so 0.0 is an unambiguous sentinel.
    #   presence_check=True tells the scaler to exclude these rows from
    #   fitting so the sentinel doesn't corrupt scaling parameters.
    #
    # is_home — ordinal flag: 1 = home, 0 = away, -1 = no fixture
    #   (blank gameweek).  Three distinct states prevent the model from
    #   conflating "away" with "didn't play".  Left unscaled (IDENTITY)
    #   as the raw values carry the full signal.  presence_check with
    #   min_value=-1 marks blank-GW rows for exclusion from scaler fits.

    FeatureSpec(
        name="team_elo",

        feature_type=FeatureType.GAUSSIAN,
        scaling_mode=ScalingMode.LINEAR,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.FIXTURE: "team_elo"},
        presence_check=True,
        min_value=0.0,
    ),
    FeatureSpec(
        name="oppo_team_elo",

        feature_type=FeatureType.GAUSSIAN,
        scaling_mode=ScalingMode.LINEAR,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.FIXTURE: "oppo_team_elo"},
        presence_check=True,
        min_value=0.0,
    ),
    FeatureSpec(
        name="is_home",

        feature_type=FeatureType.ORDINAL,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.FIXTURE: "is_home"},
        presence_check=True,
        min_value=-1,
    ),

    # ── §7  Categorical embeddings ────────────────────────────────────────
    #
    # These become x_categorical — integer indices fed to nn.Embedding.
    # They are NOT scaled by FeatureScaler (temporal=False excludes them
    # from the scaling masks).
    #
    # position           — GKP/DEF/MID/FWD (4 categories + padding 0)
    # team_code          — player's own team (20 categories + padding 0)
    # oppo_team_code    — who they're playing (20 categories + padding 0)
    #
    # Embedding dimensions follow a rough sqrt(n_categories) heuristic,
    # rounded up for power-of-2 friendliness.

    FeatureSpec(
        name="position",
        feature_type=FeatureType.CATEGORICAL,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=False,
        source_columns={DataSource.SEQUENCER: "position"},
        categories=["GKP", "DEF", "MID", "FWD"],
        embedding_dim=4,
    ),
    FeatureSpec(
        name="team_code",
        feature_type=FeatureType.CATEGORICAL,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=False,
        source_columns={DataSource.SEQUENCER: "team_code"},
        categories=team_codes,
        embedding_dim=8,
    ),
    FeatureSpec(
        name="oppo_team_code",
        feature_type=FeatureType.CATEGORICAL,
        scaling_mode=ScalingMode.IDENTITY,
        accumulation=AccumulationType.NONE,
        temporal=True,
        source_columns={DataSource.FIXTURE: "oppo_team_code"},
        categories=team_codes,
        embedding_dim=8,
    ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Convenience constructor
# ═══════════════════════════════════════════════════════════════════════════

def build_features(team_codes: list[int]) -> Features:
    """
    Build a Features instance from the canonical spec list.

    Args:
        team_codes: Integer team codes from teams.csv (e.g. [1, 2, … 20]).
            Passed through to categorical embedding specs for team_code
            and oppo_team_code.

    Returns:
        Features instance ready for the pipeline.
    """
    return Features(_build_specs(team_codes))


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity check when run directly: python feature_registry.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # default 1–20 for quick sanity checks without loading teams.csv
    features = build_features(list(range(1, 21)))

    print(f"Total features:     {len(features)}")
    print(f"  Temporal:         {len(features.temporal_columns)}")
    print(f"  Categorical:      {len(features.categorical_columns)}")
    print(f"  Snapshot:         {len(features.snapshot_columns)}")
    print(f"  Per-90:           {len(features.per_90_columns)}")
    print(f"  Raw cumulative:   {len(features.raw_cumulative_columns)}")
    print()

    print("Cumulative column mapping (output → cum):")
    for out_col, cum_col in features.cumulative_map.items():
        print(f"  {out_col:45s} → {cum_col}")
    print()

    print("Source maps:")
    for provider in [DataSource.VAASTAV, DataSource.FCI, DataSource.OPTA]:
        smap = features.source_map(provider)
        print(f"  {provider}: {len(smap)} columns")
    print()

    print("Output column order (tensor positions):")
    for i, name in enumerate(features.output_columns):
        spec = features[name]
        print(f"  [{i:2d}] {name:45s}  {spec.accumulation.value:8s}  {spec.scaling_mode.value:10s}")
