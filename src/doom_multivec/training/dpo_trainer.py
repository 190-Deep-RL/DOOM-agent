"""DPO (Direct Preference Optimization) trainer for DOOM policies.

Implements the DPO algorithm for training language model policies from
preference pairs, with frozen encoder and trainable actor head.

Reference:
    Rafailov et al. "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model". NeurIPS 2023.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.dpo_policy import DPODoomPolicy


class DPOTrainer:
    """Trainer for Direct Preference Optimization.

    Args:
        policy: Trainable DPODoomPolicy.
        ref_policy: Frozen reference policy (copy of initial policy).
        beta: DPO temperature parameter (default 0.1).
              Lower = more divergence from reference allowed.
        lr: Learning rate (default 5e-5).
        weight_decay: Weight decay for AdamW (default 0.01).
        device: Device to train on.
    """

    def __init__(
        self,
        policy: DPODoomPolicy,
        ref_policy: DPODoomPolicy,
        beta: float = 0.1,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        device: str = 'cuda',
    ):
        self.policy = policy.to(device)
        self.ref_policy = ref_policy.to(device)
        self.beta = beta
        self.device = device

        # Freeze reference policy
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()

        # Optimizer applies to policy.actor_head since encoder is frozen
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),  # Only actor_head has grads, but this is simpler
            lr=lr,
            weight_decay=weight_decay,
        )

        self.global_step = 0
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'eval_metrics': [],
        }

    def compute_dpo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        win_actions: torch.Tensor,
        lose_actions: torch.Tensor,
        depth_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss for a batch.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            win_actions: (batch,) preferred action indices.
            lose_actions: (batch,) dispreferred action indices.
            depth_ids: Optional (batch, seq_len) depth bin IDs.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size = input_ids.shape[0]

        # Policy log-probs for win and lose actions
        policy_logprob_win = self.policy.get_logprobs(
            input_ids, attention_mask, win_actions, depth_ids
        )
        policy_logprob_lose = self.policy.get_logprobs(
            input_ids, attention_mask, lose_actions, depth_ids
        )

        # Reference policy log-probs (no grad)
        with torch.no_grad():
            ref_logprob_win = self.ref_policy.get_logprobs(
                input_ids, attention_mask, win_actions, depth_ids
            )
            ref_logprob_lose = self.ref_policy.get_logprobs(
                input_ids, attention_mask, lose_actions, depth_ids
            )

        # DPO log-ratios
        policy_ratio_win = policy_logprob_win - ref_logprob_win
        policy_ratio_lose = policy_logprob_lose - ref_logprob_lose

        # DPO loss: -log(sigmoid(beta * (ratio_win - ratio_lose)))
        logits = self.beta * (policy_ratio_win - policy_ratio_lose)
        loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        with torch.no_grad():
            # Preference accuracy: how often policy ranks winner higher
            accuracy = (logits > 0).float().mean().item()

            # Average log-prob ratios
            avg_win_ratio = policy_ratio_win.mean().item()
            avg_lose_ratio = policy_ratio_lose.mean().item()

            # KL divergence estimate from reference
            kl_div = (policy_ratio_win - 1).mean().item()  # Approximation

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'avg_win_ratio': avg_win_ratio,
            'avg_lose_ratio': avg_lose_ratio,
            'kl_div': kl_div,
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
        }

        return loss, metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_every: int = 50,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.
            log_every: Log metrics every N steps.

        Returns:
            Dict of average metrics for the epoch.
        """
        self.policy.train()

        epoch_metrics = {
            'loss': [],
            'accuracy': [],
            'avg_win_ratio': [],
            'avg_lose_ratio': [],
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            win_actions = batch['win_action'].to(self.device)
            lose_actions = batch['lose_action'].to(self.device)
            depth_ids = batch.get('depth_ids')
            if depth_ids is not None:
                depth_ids = depth_ids.to(self.device)

            # Compute loss
            loss, metrics = self.compute_dpo_loss(
                input_ids, attention_mask, win_actions, lose_actions, depth_ids
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.global_step += 1

            # Collect metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.2%}",
            })

            # Log periodically
            if (step + 1) % log_every == 0:
                avg_loss = np.mean(epoch_metrics['loss'][-log_every:])
                avg_acc = np.mean(epoch_metrics['accuracy'][-log_every:])
                print(f"  Step {self.global_step}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

        # Compute epoch averages
        epoch_summary = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }

        self.history['train_loss'].extend(epoch_metrics['loss'])
        self.history['train_accuracy'].extend(epoch_metrics['accuracy'])

        return epoch_summary

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set.

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict of evaluation metrics.
        """
        self.policy.eval()

        all_metrics = {
            'loss': [],
            'accuracy': [],
            'avg_win_ratio': [],
            'avg_lose_ratio': [],
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                win_actions = batch['win_action'].to(self.device)
                lose_actions = batch['lose_action'].to(self.device)
                depth_ids = batch.get('depth_ids')
                if depth_ids is not None:
                    depth_ids = depth_ids.to(self.device)

                loss, metrics = self.compute_dpo_loss(
                    input_ids, attention_mask, win_actions, lose_actions, depth_ids
                )

                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(metrics[key])

        eval_summary = {
            f'eval_{key}': np.mean(values) for key, values in all_metrics.items()
        }

        self.history['eval_metrics'].append(eval_summary)

        return eval_summary

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        num_epochs: int = 3,
        eval_every: int = 1,
        save_path: str | None = None,
        save_every: int = 1,
    ) -> dict[str, Any]:
        """Full training loop.

        Args:
            train_dataloader: Training data.
            eval_dataloader: Optional validation data.
            num_epochs: Number of epochs to train.
            eval_every: Evaluate every N epochs.
            save_path: Optional path to save checkpoints.
            save_every: Save checkpoint every N epochs.

        Returns:
            Training history dict.
        """
        print("=" * 60)
        print("DPO Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Beta: {self.beta}")
        print(f"Epochs: {num_epochs}")
        print(f"Steps per epoch: {len(train_dataloader)}")

        # Print parameter counts
        param_counts = self.policy.count_parameters()
        print(f"\nModel parameters:")
        print(f"  Total: {param_counts['total']:,}")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Frozen: {param_counts['frozen']:,}")
        print(f"  Actor head: {param_counts['actor_head']:,}")

        best_eval_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 40)

            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)
            print(f"Train metrics: loss={train_metrics['loss']:.4f}, "
                  f"acc={train_metrics['accuracy']:.2%}")

            # Evaluate
            if eval_dataloader is not None and epoch % eval_every == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                print(f"Eval metrics: loss={eval_metrics['eval_loss']:.4f}, "
                      f"acc={eval_metrics['eval_accuracy']:.2%}")

                # Track best model
                if eval_metrics['eval_accuracy'] > best_eval_acc:
                    best_eval_acc = eval_metrics['eval_accuracy']
                    if save_path:
                        best_path = os.path.join(save_path, 'best')
                        self.save_checkpoint(best_path, is_best=True)
                        print(f"  New best model saved (acc={best_eval_acc:.2%})")

            # Save checkpoint
            if save_path and epoch % save_every == 0:
                checkpoint_path = os.path.join(save_path, f'checkpoint-epoch-{epoch}')
                self.save_checkpoint(checkpoint_path)
                print(f"  Checkpoint saved to {checkpoint_path}")

        # Save final model
        if save_path:
            final_path = os.path.join(save_path, 'final')
            self.save_checkpoint(final_path)
            print(f"\nFinal model saved to {final_path}")

        print("=" * 60)
        print("Training complete!")
        print(f"Best eval accuracy: {best_eval_acc:.2%}")

        return self.history

    def save_checkpoint(
        self,
        save_path: str,
        is_best: bool = False,
    ) -> None:
        """Save training checkpoint.

        Args:
            save_path: Directory to save to.
            is_best: Whether this is the best model so far.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save policy
        self.policy.save_pretrained(save_path, save_encoder=True)

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))

        # Save training state
        state = {
            'global_step': self.global_step,
            'beta': self.beta,
            'history': self.history,
            'is_best': is_best,
        }
        with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        # Load policy
        self.policy = DPODoomPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)

        # Load optimizer
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        # Load training state
        state_path = os.path.join(checkpoint_path, 'training_state.json')
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state.get('global_step', 0)
            self.history = state.get('history', self.history)


def create_reference_policy(policy: DPODoomPolicy) -> DPODoomPolicy:
    """Create a frozen copy of a policy as reference.

    Args:
        policy: Policy to copy.

    Returns:
        Frozen copy of the policy.
    """
    # Create new instance with same config
    ref_policy = DPODoomPolicy(
        encoder_path=policy.encoder.config._name_or_path,
        hidden_size=policy.hidden_size,
        actor_hidden=policy.actor_head[0].out_features,
        num_actions=policy.num_actions,
        freeze_encoder=True,
    )

    # Copy weights
    ref_policy.load_state_dict(policy.state_dict())

    # Freeze everything
    for param in ref_policy.parameters():
        param.requires_grad = False
    ref_policy.eval()

    return ref_policy
