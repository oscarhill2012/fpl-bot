from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Model output
# ============================================================================


@dataclass
class ModelOutput:
    """
    Structured output from the FPL predictor.

    Contains the final points prediction (used by the loss function)
    alongside the individual component predictions (used for
    interpretability and diagnostics at inference time).

    Attributes:
        points: Predicted total FPL points.  Shape ``[B, K]``.
        minutes_probs: Softmax probabilities over minutes-bracket
            classes (no play / partial / full).  Shape ``[B, K, 3]``.
        components: Activated regression outputs for each scoring
            event, ordered by ``ScoringRules`` column indices:
            goals, assists, clean_sheets, goals_conceded, saves,
            bonus.  Shape ``[B, K, 6]``.
    """

    points: torch.Tensor
    minutes_probs: torch.Tensor
    components: torch.Tensor


# ============================================================================
# Scoring rules
# ============================================================================


class ScoringRules(nn.Module):
    """
    Non-learnable module that combines component predictions into
    FPL points using the official scoring matrix.

    All tensors are registered as buffers so they follow the model
    across devices but never receive optimiser updates.  The forward
    pass is fully differentiable — gradients flow through the matrix
    multiply and softmax back to each component head.

    Component column indices (constants on the class):

    ======  =====
    Name    Index
    ======  =====
    GOALS       0
    ASSISTS     1
    CS          2
    GC          3
    SAVES       4
    BONUS       5
    ======  =====

    FPL position IDs are 1-indexed (1 = GK, 2 = DEF, 3 = MID,
    4 = FWD).  The scoring matrix is 0-indexed, so we subtract 1
    when looking up weights.
    """

    # --- Component column indices ---
    GOALS = 0
    ASSISTS = 1
    CLEAN_SHEETS = 2
    GOALS_CONCEDED = 3
    SAVES = 4
    BONUS = 5

    N_COMPONENTS = 6
    N_MINUTES_CLASSES = 3

    def __init__(self) -> None:
        """Register the FPL scoring matrix and appearance points as buffers."""
        super().__init__()

        # Scoring matrix: [4 positions, 6 components].
        # Rows:  GK (0), DEF (1), MID (2), FWD (3).
        # Cols:  goals, assists, CS, GC, saves, bonus.
        #
        # Goals conceded uses the fractional rate (-0.5 per goal)
        # rather than the floor rule (-1 per 2 goals).  For
        # continuous predictions this gives the correct expected
        # value; at integer-rounded inference the approximation
        # error is negligible.
        #
        # Saves uses 1/3 per save (1 point per 3 saves), same
        # fractional reasoning.
        self.register_buffer(
            "scoring_matrix",
            torch.tensor([
                # goals  assists   CS     GC     saves  bonus
                [10.0,   3.0,     4.0,  -0.5,   1/3,   1.0],   # GK
                [ 6.0,   3.0,     4.0,  -0.5,   0.0,   1.0],   # DEF
                [ 5.0,   3.0,     1.0,   0.0,   0.0,   1.0],   # MID
                [ 4.0,   3.0,     0.0,   0.0,   0.0,   1.0],   # FWD
            ]),
        )

        # Appearance points per minutes-bracket class.
        # Class 0 = did not play,  class 1 = 1-59 min,  class 2 = 60+.
        self.register_buffer(
            "appearance_points",
            torch.tensor([0.0, 1.0, 2.0]),
        )

        logger.info(
            "ScoringRules: %d components, %d minutes classes, "
            "%d positions",
            self.N_COMPONENTS, self.N_MINUTES_CLASSES,
            self.scoring_matrix.shape[0],
        )

    # =====================================================
    # Forward Pass
    # =====================================================

    def forward(
        self,
        minutes_probs: torch.Tensor,
        components: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine component predictions into total FPL points.

        All operations are differentiable — gradients propagate
        back through the fixed scoring weights to each head.

        Args:
            minutes_probs: Softmax probabilities for the three
                minutes-bracket classes.  Shape ``[B, K, 3]``.
            components: Activated regression outputs (goals,
                assists, CS, GC, saves, bonus).  Shape ``[B, K, 6]``.
            position_id: Raw FPL position ID per player
                (1 = GK, 2 = DEF, 3 = MID, 4 = FWD).
                Shape ``[B]``.

        Returns:
            Predicted total points.  Shape ``[B, K]``.
        """
        # --- Appearance points ---
        # Expected value over the three minutes-bracket classes.
        # [B, K, 3] · [3] → [B, K]
        appearance = torch.matmul(
            minutes_probs, self.appearance_points,
        )

        # --- Component points ---
        # Look up the scoring-weight row for each player's position.
        pos_idx = position_id - 1                       # [B]
        weights = self.scoring_matrix[pos_idx]           # [B, 6]

        # Broadcast weights across the predict window K.
        weights = weights.unsqueeze(1).expand_as(        # [B, K, 6]
            components,
        )

        # Element-wise multiply and sum over components.
        component_points = (components * weights).sum(   # [B, K]
            dim=-1,
        )

        return appearance + component_points

    # =====================================================
    # Component Activation
    # =====================================================

    @staticmethod
    def activate_components(raw: torch.Tensor) -> torch.Tensor:
        """
        Apply per-component activations to raw regression outputs.

        Enforces physical constraints on each scoring event:

        * Goals, assists, goals conceded, saves, bonus — ReLU
          (non-negative counts).
        * Clean sheets — sigmoid (probability bounded to [0, 1]).

        The clean-sheet sigmoid is important because the scoring
        weight can be 4× (for DEF/GK).  Without bounding, the model
        could predict CS > 1 and produce inflated points that mask
        errors in other components.

        Args:
            raw: Raw linear outputs from the regression head.
                Shape ``[..., 6]``.

        Returns:
            Activated outputs, same shape.
        """
        cs = ScoringRules.CLEAN_SHEETS
        return torch.cat([
            torch.relu(raw[..., :cs]),          # goals, assists
            torch.sigmoid(raw[..., cs:cs + 1]), # clean_sheets
            torch.relu(raw[..., cs + 1:]),      # GC, saves, bonus
        ], dim=-1)

    # =====================================================
    # Diagnostic Helpers
    # =====================================================

    @staticmethod
    def component_names() -> list[str]:
        """Return component names in column order."""
        return [
            "goals", "assists", "clean_sheets",
            "goals_conceded", "saves", "bonus",
        ]

    @staticmethod
    def decompose(output: ModelOutput) -> dict[str, torch.Tensor]:
        """
        Split a ModelOutput into named component tensors.

        Useful at inference time for inspecting what the model
        thinks will happen per scoring event.

        Args:
            output: A ModelOutput from the predictor.

        Returns:
            Dict mapping component name to its prediction tensor
            (shape ``[B, K]`` for each).  Also includes
            ``minutes_probs`` with shape ``[B, K, 3]``.
        """
        names = ScoringRules.component_names()
        result: dict[str, torch.Tensor] = {
            name: output.components[..., i]
            for i, name in enumerate(names)
        }
        result["minutes_probs"] = output.minutes_probs
        return result
