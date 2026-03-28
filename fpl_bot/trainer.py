from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm.auto import tqdm

from .model import FPLPointsPredictor
from .pipeline import FeatureScaler

logger = logging.getLogger(__name__)


# ============================================================================
# Training history
# ============================================================================


@dataclass
class TrainHistory:
    """
    Container for per-epoch training metrics.

    Attributes:
        train_loss: Mean training MSE per epoch.
        val_loss: Mean validation MSE per epoch.
        val_mae: Mean validation MAE per epoch (more interpretable ---
            "predictions are off by X points on average").
        lr: Learning rate at the end of each epoch.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    grad_norm: list[float] = field(default_factory=list)
    pred_mean: list[float] = field(default_factory=list)
    pred_std: list[float] = field(default_factory=list)
    val_mae_by_horizon: list[list[float]] = field(default_factory=list)


# ============================================================================
# Trainer
# ============================================================================


class Trainer:
    """
    Manages training and validation of an FPLPointsPredictor.

    Owns the optimiser, loss function, scheduler, and training loop.
    The fitted ``FeatureScaler`` is applied per-batch during forward
    passes so the ``DataLoader`` yields raw data throughout.

    Args:
        model: The model to train.
        scaler: A *fitted* FeatureScaler (``train_scale`` already called).
        train_loader: DataLoader yielding training batches.
        val_loader: DataLoader yielding validation batches.
        lr: Initial learning rate for Adam.
        weight_decay: L2 penalty coefficient.  Set to 0 to disable.
        grad_clip: Maximum gradient norm.  Gradients exceeding this
            are rescaled proportionally.
        device: Torch device.  Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: FPLPointsPredictor,
        scaler: FeatureScaler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ):
        """Initialise trainer with model, data, and hyperparameters."""
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model = model.to(self.device)
        self.scaler = scaler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.grad_clip = grad_clip

        self.criterion = nn.MSELoss()
        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=5,
        )

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Trainer initialised: %d params, device=%s, lr=%.1e, "
            "grad_clip=%.1f, weight_decay=%.1e",
            param_count, self.device, lr, grad_clip, weight_decay,
        )

    #=====================================================
    # Public Interface
    #=====================================================

    def fit(
        self,
        epochs: int,
        patience: int = 10,
        run_version: str = "default",
    ) -> TrainHistory:
        """
        Run the full training loop with early stopping.

        Args:
            epochs: Maximum number of training epochs.
            patience: Stop training after this many epochs without
                validation loss improvement.
            run_version: Subdirectory name under ``model_performance/``
                for this run's outputs (checkpoints, dashboard).

        Returns:
            TrainHistory with per-epoch metrics.
        """
        output_dir = self._resolve_output_dir(run_version)
        history = TrainHistory()
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        logger.info(
            "Starting training: %d epochs, patience=%d, "
            "output_dir=%s",
            epochs, patience, output_dir,
        )

        history = TrainHistory()
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in epoch_bar:
            train_loss, grad_norm = self._train_epoch()
            val_loss, val_mae, val_mae_by_horizon, pred_mean, pred_std = self._validate()
            current_lr = self.optimiser.param_groups[0]["lr"]

            self.scheduler.step(val_loss)

            # Record metrics
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_mae.append(val_mae)
            history.lr.append(current_lr)

            history.grad_norm.append(grad_norm)
            history.pred_mean.append(pred_mean)
            history.pred_std.append(pred_std)
            history.val_mae_by_horizon.append(val_mae_by_horizon)

            # Update progress bar
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                mae=f"{val_mae:.2f}",
                lr=f"{current_lr:.1e}",
            )

            # Populate TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("MAE/val", val_mae, epoch)
            writer.add_scalar("LR", current_lr, epoch)

            writer.add_scalar("Train/grad_norm", grad_norm, epoch)
            writer.add_scalar("Pred/mean", pred_mean, epoch)
            writer.add_scalar("Pred/std", pred_std, epoch)

            for k, mae_k in enumerate(val_mae_by_horizon):
                writer.add_scalar(f"MAE/GW+{k+1}", mae_k, epoch)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(
                    output_dir / "best_model.pt", epoch, best_val_loss,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d — no improvement "
                    "for %d epochs. Best val loss: %.4f",
                    epoch, patience, best_val_loss,
                )
                break

        # Save final checkpoint regardless of early stopping
        self.save_checkpoint(
            output_dir / "final_model.pt", epoch, val_loss,
        )

        logger.info(
            "Training complete: %d epochs, best val loss %.4f, "
            "final val MAE %.2f",
            epoch + 1, best_val_loss, history.val_mae[-1],
        )

        writer.close()
        self._save_training_summary_plot(output_dir, history, best_val_loss)
        self._save_history_json(output_dir / "training_history.json", history)

        return history

    #=====================================================
    # Checkpointing
    # Save and restore model, optimiser, and scheduler.
    #=====================================================

    def save_checkpoint(
        self,
        path: pathlib.Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            val_loss: Validation loss at this epoch.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )
        logger.debug("Checkpoint saved to %s (epoch %d)", path, epoch)

    def load_checkpoint(self, path: pathlib.Path) -> dict:
        """
        Load a training checkpoint and restore model/optimiser state.

        Args:
            path: Path to a saved checkpoint file.

        Returns:
            The full checkpoint dict (includes epoch and val_loss).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        # weights_only=False required because the checkpoint includes
        # optimiser and scheduler state (Python dicts, not just tensors)
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(
            checkpoint["optimiser_state_dict"],
        )
        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"],
        )

        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss %.4f)",
            path, checkpoint["epoch"], checkpoint["val_loss"],
        )
        return checkpoint

    #=====================================================
    # Private Helpers
    #=====================================================

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch and return mean batch loss."""
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        
        for batch in self.train_loader:
            # Scale on CPU (scaler params are CPU tensors), then move
            x_scaled = self.scaler.test_scale(
                batch["x_numeric"],
            ).to(self.device)

            x_cat = batch["x_categorical"].to(self.device)
            x_fix = batch["x_future_fixtures"].to(self.device)
            y = batch["y"].to(self.device)

            pred = self.model(x_scaled, x_cat, x_fix)
            loss = self.criterion(pred, y)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip,
            )
            self.optimiser.step()

            total_grad_norm += float(grad_norm)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches, total_grad_norm / n_batches

    def _validate(self) -> tuple[float, float, list[float], float, float]:
        """Run validation and return (mean_loss, mean_mae)."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0
        total_abs_error_by_horizon = None
        pred_sum = 0.0
        pred_sq_sum = 0.0
        pred_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x_scaled = self.scaler.test_scale(
                    batch["x_numeric"],
                ).to(self.device)

                x_cat = batch["x_categorical"].to(self.device)
                x_fix = batch["x_future_fixtures"].to(self.device)
                y = batch["y"].to(self.device)

                pred = self.model(x_scaled, x_cat, x_fix)

                abs_err = torch.abs(pred - y)
                horizon_abs = abs_err.sum(dim=0).detach().cpu()
                if total_abs_error_by_horizon is None:
                    total_abs_error_by_horizon = horizon_abs.clone()
                else:
                    total_abs_error_by_horizon += horizon_abs

               
                pred_sum += pred.sum().item()
                pred_sq_sum += (pred ** 2).sum().item()
                pred_count += pred.numel()
                total_loss += self.criterion(pred, y).item()
                total_mae += torch.mean(torch.abs(pred - y)).item()
                n_batches += 1

        mean_loss = total_loss / n_batches
        mean_mae = total_mae / n_batches

        mae_by_horizon = (total_abs_error_by_horizon / len(self.val_loader.dataset)).tolist()

        pred_mean = pred_sum / pred_count
        pred_var = max((pred_sq_sum / pred_count) - pred_mean**2, 0.0)
        pred_std = pred_var ** 0.5

        return mean_loss, mean_mae, mae_by_horizon, pred_mean, pred_std

    def _resolve_output_dir(self, run_version: str) -> pathlib.Path:
        """Resolve the run output directory, creating it if needed."""
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "model_performance" / run_version
        )
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _save_history_json(self, path: pathlib.Path, history: TrainHistory) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history.__dict__, f, indent=2)


    def _save_training_summary_plot(
        self,
        run_dir: pathlib.Path,
        history: TrainHistory,
        best_val_loss: float,
    ) -> None:
        epochs = range(1, len(history.train_loss) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, history.train_loss, label="train")
        ax.plot(epochs, history.val_loss, label="val")
        ax.axhline(best_val_loss, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 10)
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # MAE
        ax = axes[0, 1]
        ax.plot(epochs, history.val_mae)
        ax.set_title("Validation MAE")
        ax.grid(True, alpha=0.3)

        # LR
        ax = axes[1, 0]
        ax.plot(epochs, history.lr)
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)

        # Generalisation gap
        ax = axes[1, 1]
        gap = [v - t for t, v in zip(history.train_loss, history.val_loss)]
        ax.plot(epochs, gap)
        ax.set_ylim(-10, 10)
        ax.set_title("Generalisation Gap")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(run_dir / "training_summary.png", dpi=200)
        plt.close(fig)

        # Horizon MAE
        if history.val_mae_by_horizon:
            final_mae = history.val_mae_by_horizon[-1]
            fig, ax = plt.subplots()
            ax.bar([f"GW+{i+1}" for i in range(len(final_mae))], final_mae)
            ax.set_title("Final MAE by Horizon")
            fig.savefig(run_dir / "final_horizon_mae.png", dpi=200)
            plt.close(fig)
