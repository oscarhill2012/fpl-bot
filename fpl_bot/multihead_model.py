from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .features import Features
from .scoring import ModelOutput, ScoringRules

logger = logging.getLogger(__name__)


class FPLPointsPredictorMH(nn.Module):
    """
    LSTM encoder + multi-head MLP decoder for FPL points prediction.

    The encoder compresses the gameweek sequence into a context vector
    (unchanged from the single-head baseline).  The decoder fans out
    into two heads — a 3-class minutes-bracket classifier and a
    6-component regression head — whose outputs are recombined by a
    frozen ``ScoringRules`` layer using the official FPL scoring matrix.

    Because the scoring rules layer is fully differentiable, the loss
    is computed on total predicted points (identical to the baseline),
    and gradients flow back through the scoring matrix to train each
    component head.  At inference time the component predictions are
    available for interpretability.

    Args:
        n_numeric_features: Number of scaled numeric input features.
        categorical_vocab_sizes: Vocabulary size per categorical
            feature, including the padding index.  Position has 4
            real categories + 1 padding = 5.
        categorical_embedding_dims: Embedding dimensionality per
            categorical feature.  Must match the length of
            ``categorical_vocab_sizes``.
        n_fixture_features: Features per target gameweek in
            ``x_future_fixtures`` (team_elo, oppo_team_elo, is_home,
            num_matches, team_code = 5).
        lstm_hidden_dim: LSTM hidden state size.
        lstm_layers: Number of stacked LSTM layers.
        mlp_hidden_dim: Width of the shared decoder trunk.
        dropout: Dropout probability for LSTM and MLP layers.
    """

    def __init__(
        self,
        n_numeric_features: int,
        categorical_vocab_sizes: list[int],
        categorical_embedding_dims: list[int],
        n_fixture_features: int,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        """Initialise model layers and apply weight initialisation."""
        super().__init__()

        if len(categorical_vocab_sizes) != len(categorical_embedding_dims):
            raise ValueError(
                "categorical_vocab_sizes and "
                "categorical_embedding_dims must have the same length."
            )

        # Store configuration for serialisation and inspection.
        self.n_numeric_features = n_numeric_features
        self.n_fixture_features = n_fixture_features
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        #=====================================================
        # Embedding Layers
        # One nn.Embedding per categorical feature. Index 0 is
        # reserved as padding and always returns a zero vector.
        #=====================================================

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab, dim, padding_idx=0)
            for vocab, dim in zip(
                categorical_vocab_sizes,
                categorical_embedding_dims,
            )
        ])
        total_embedding_dim = sum(categorical_embedding_dims)

        #=====================================================
        # LSTM Encoder
        # Reads the gameweek sequence and compresses it into a
        # fixed-size context vector (final hidden state).
        #=====================================================

        lstm_input_dim = n_numeric_features + total_embedding_dim
        self.encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        #=====================================================
        # Context Normalisation
        # LayerNorm stabilises the LSTM output before the MLP.
        #=====================================================

        self.context_norm = nn.LayerNorm(lstm_hidden_dim)

        #=====================================================
        # Multi-Head Decoder
        # Shared trunk extracts a hidden representation from the
        # context vector + fixture features.  Two heads project
        # into scoring components:
        #   - minutes_head: 3-class logits (no play / partial / full)
        #   - regression_head: 6 scoring events (goals, assists,
        #     CS, GC, saves, bonus)
        #=====================================================

        decoder_input_dim = lstm_hidden_dim + n_fixture_features
        self.decoder_trunk = nn.Sequential(
            nn.Linear(decoder_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.minutes_head = nn.Linear(
            mlp_hidden_dim // 2, ScoringRules.N_MINUTES_CLASSES,
        )
        self.regression_head = nn.Linear(
            mlp_hidden_dim // 2, ScoringRules.N_COMPONENTS,
        )

        #=====================================================
        # Scoring Rules (frozen)
        # Differentiable but non-learnable layer that applies the
        # FPL scoring matrix to component predictions.
        #=====================================================

        self.scoring_rules = ScoringRules()

        self._initialise_weights()

        # Log architecture summary for debugging.
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            "FPLPointsPredictor: LSTM(%d -> %d, %d layers) "
            "+ trunk(%d -> %d -> %d) "
            "+ heads(minutes=%d, regression=%d), %d params",
            lstm_input_dim, lstm_hidden_dim, lstm_layers,
            decoder_input_dim, mlp_hidden_dim, mlp_hidden_dim // 2,
            ScoringRules.N_MINUTES_CLASSES, ScoringRules.N_COMPONENTS,
            param_count,
        )

    #=====================================================
    # Factory Methods
    # Construct the model from project-level objects.
    #=====================================================

    @classmethod
    def from_features(
        cls,
        features: Features,
        n_fixture_features: int = 5,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> FPLPointsPredictorMH:
        """
        Construct the model from a Features registry.

        Extracts numeric feature count, categorical vocabulary sizes,
        and embedding dimensions directly from the registry.

        Args:
            features: Feature registry describing all model inputs.
            n_fixture_features: Fixture features per target gameweek.
                Defaults to 5 (team_elo, oppo_team_elo, is_home,
                num_matches, team_code).
            lstm_hidden_dim: LSTM hidden state size.
            lstm_layers: Number of stacked LSTM layers.
            mlp_hidden_dim: MLP hidden layer width.
            dropout: Dropout probability.

        Returns:
            Configured FPLPointsPredictor instance.
        """
        n_numeric = len(features.numeric_columns)

        cat_specs = [
            features[name]
            for name in features.categorical_columns
        ]

        # Vocab size = highest possible index + 1 (slot 0 is padding).
        # Integer categories (e.g. team_code) use raw values as indices,
        # so we need max(codes) + 1.  String categories (e.g. position)
        # are mapped to sequential 1..N, so len + 1 suffices.
        vocab_sizes = []
        for spec in cat_specs:
            if all(isinstance(c, int) for c in spec.categories):
                vocab_sizes.append(max(spec.categories) + 1)
            else:
                vocab_sizes.append(len(spec.categories) + 1)

        embedding_dims = [
            spec.embedding_dim for spec in cat_specs
        ]

        logger.info(
            "from_features: %d numeric, %d categorical (%s), "
            "%d fixture features",
            n_numeric,
            len(cat_specs),
            ", ".join(
                f"{s.name}({v} -> {d}d)"
                for s, v, d in zip(
                    cat_specs, vocab_sizes, embedding_dims,
                )
            ),
            n_fixture_features,
        )

        return cls(
            n_numeric_features=n_numeric,
            categorical_vocab_sizes=vocab_sizes,
            categorical_embedding_dims=embedding_dims,
            n_fixture_features=n_fixture_features,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

    #=====================================================
    # Forward Pass
    # Core inference logic.
    #=====================================================

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        x_future_fixtures: torch.Tensor,
        position_id: torch.Tensor,
    ) -> ModelOutput:
        """
        Run the full encoder → multi-head decoder → scoring rules
        forward pass.

        Args:
            x_numeric: Scaled numeric features.
                Shape ``[B, T, F]``.
            x_categorical: Integer category indices.
                Shape ``[B, T, C]``.
            x_future_fixtures: Fixture context for target gameweeks.
                Shape ``[B, K, fix_feat]``.
            position_id: Raw FPL position ID per player
                (1 = GK, 2 = DEF, 3 = MID, 4 = FWD).
                Shape ``[B]``.

        Returns:
            ModelOutput containing predicted points ``[B, K]``,
            minutes class probabilities ``[B, K, 3]``, and
            activated component predictions ``[B, K, 6]``.
        """
        batch_size = x_numeric.shape[0]
        predict_window = x_future_fixtures.shape[1]

        # --- Embed categorical features ---
        # Each embedding layer maps one column of integer indices to a
        # dense vector.  Results are concatenated along the feature
        # dimension to form the full embedded representation.
        embedded = [
            emb(x_categorical[:, :, i])
            for i, emb in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(embedded, dim=-1)  # [B, T, E]

        # --- Encode the gameweek sequence ---
        # Concatenate numeric features and embeddings, then pass
        # through the LSTM.  The final hidden state of the last
        # layer becomes the context vector.
        lstm_input = torch.cat(
            [x_numeric, x_emb], dim=-1,
        )  # [B, T, F + E]
        _, (h_n, _) = self.encoder(lstm_input)

        # h_n shape: [num_layers, B, H].  Take the last layer and
        # normalise to stabilise scale before the MLP.
        context = self.context_norm(h_n[-1])  # [B, H]

        # --- Decode: predict components per target gameweek ---
        # The trunk and heads are shared across all K target GWs
        # (parameter sharing).  Expand the context to pair with each
        # fixture row, then batch all K predictions in one pass.
        fixtures = x_future_fixtures.float()  # [B, K, fix]

        context_expanded = context.unsqueeze(1).expand(
            -1, predict_window, -1,
        )  # [B, K, H]

        mlp_input = torch.cat(
            [context_expanded, fixtures], dim=-1,
        )  # [B, K, H + fix]

        # Flatten K into the batch dimension for a single trunk pass.
        flat = mlp_input.reshape(
            batch_size * predict_window, -1,
        )  # [B*K, H + fix]

        trunk_out = self.decoder_trunk(flat)         # [B*K, hidden//2]
        minutes_logits = self.minutes_head(trunk_out) # [B*K, 3]
        regression_raw = self.regression_head(trunk_out)  # [B*K, 6]

        # Reshape back to [B, K, ...].
        minutes_logits = minutes_logits.reshape(
            batch_size, predict_window, ScoringRules.N_MINUTES_CLASSES,
        )
        regression_raw = regression_raw.reshape(
            batch_size, predict_window, ScoringRules.N_COMPONENTS,
        )

        # --- Activate component predictions ---
        # Softmax for minutes (class probabilities), sigmoid for
        # clean sheets (bounded 0-1), ReLU for counts (non-negative).
        minutes_probs = torch.softmax(minutes_logits, dim=-1)
        components = ScoringRules.activate_components(regression_raw)

        # --- Apply scoring rules ---
        # Differentiable combination using the FPL scoring matrix.
        # Gradients flow back through this layer to each head.
        points = self.scoring_rules(
            minutes_probs, components, position_id,
        )

        return ModelOutput(
            points=points,
            minutes_probs=minutes_probs,
            components=components,
        )

    #=====================================================
    # Inference Helpers
    # Convenience methods for evaluation and prediction.
    #=====================================================

    @torch.no_grad()
    def predict(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        x_future_fixtures: torch.Tensor,
        position_id: torch.Tensor,
    ) -> ModelOutput:
        """
        Predict points and components with gradients disabled.

        Sets the model to evaluation mode (disabling dropout), runs
        the forward pass without gradient tracking, then restores the
        previous training mode.

        Args:
            x_numeric: Scaled numeric features, ``[B, T, F]``.
            x_categorical: Category indices, ``[B, T, C]``.
            x_future_fixtures: Fixture context, ``[B, K, fix]``.
            position_id: FPL position IDs, ``[B]``.

        Returns:
            ModelOutput with points ``[B, K]``, minutes class
            probabilities ``[B, K, 3]``, and component predictions
            ``[B, K, 6]``.
        """
        was_training = self.training
        self.eval()
        output = self(
            x_numeric, x_categorical,
            x_future_fixtures, position_id,
        )
        if was_training:
            self.train()
        return output

    #=====================================================
    # Private Helpers
    #=====================================================

    def _initialise_weights(self) -> None:
        """Apply Xavier init to linear layers, orthogonal init to LSTM."""
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in (
            *self.decoder_trunk.modules(),
            self.minutes_head,
            self.regression_head,
        ):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
