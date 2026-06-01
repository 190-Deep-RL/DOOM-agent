"""DPO-compatible policy model for DOOM MultiVec.

Combines a frozen ModernBERT-Hash encoder with a small trainable actor head
for Direct Preference Optimization. The encoder provides state representations
from ASCII frames; the actor head maps these to action preferences.

Architecture:
    ASCII Frame → [Frozen Encoder] → [CLS] embedding (128-d)
                                 ↓
                         [Trainable Actor Head]
                         Linear(128, 64) → GELU → Linear(64, 6)
                                 ↓
                          Action Logits (6 actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DPODoomPolicy(nn.Module):
    """DPO policy with frozen encoder and trainable actor head.

    Args:
        encoder_path: Path to pretrained ModernBERT-Hash model.
        hidden_size: Encoder output dimension (default 128).
        actor_hidden: Actor head hidden layer size (default 64).
        num_actions: Number of action classes (default 6).
        freeze_encoder: Whether to freeze encoder weights (default True).
    """

    ACTION_NAMES = [
        'shoot', 'move_forward', 'turn_left',
        'turn_right', 'strafe_left', 'strafe_right',
    ]

    def __init__(
        self,
        encoder_path: str,
        hidden_size: int = 128,
        actor_hidden: int = 64,
        num_actions: int = 6,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # Load encoder
        self.encoder = AutoModel.from_pretrained(
            encoder_path, trust_remote_code=True
        )
        self.hidden_size = hidden_size

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Small actor head: hidden_size -> actor_hidden -> num_actions
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, actor_hidden),
            nn.GELU(),
            nn.Linear(actor_hidden, num_actions),
        )

        # Initialize actor head conservatively (near zero)
        self._init_actor_head()

        self.num_actions = num_actions

    def _init_actor_head(self):
        """Initialize actor head with small weights for conservative start."""
        for module in self.actor_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_ids: torch.Tensor | None = None,
        return_logprobs: bool = False,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            depth_ids: Optional (batch, seq_len) depth bin IDs.
            return_logprobs: If True, also return log-probabilities of actions.
            actions: (batch,) action indices for log-prob computation.

        Returns:
            Dict with 'logits' and optionally 'logprobs'.
        """
        # Encode inputs (no grad if frozen)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth_ids=depth_ids,
        )

        # Use [CLS] token (index 0) as state representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        # Actor head produces action logits
        logits = self.actor_head(cls_embedding)  # (batch, num_actions)

        result = {'logits': logits}

        if return_logprobs and actions is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather log-prob of taken actions
            action_logprobs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            result['logprobs'] = action_logprobs

        return result

    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        actions: torch.Tensor,
        depth_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get log-probabilities of given actions.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) attention mask.
            actions: (batch,) action indices.
            depth_ids: Optional (batch, seq_len) depth bin IDs.

        Returns:
            (batch,) log-probabilities.
        """
        result = self.forward(
            input_ids, attention_mask, depth_ids,
            return_logprobs=True, actions=actions
        )
        return result['logprobs']

    def sample_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_ids: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy.

        Args:
            input_ids: (batch, seq_len) or (seq_len,) token IDs.
            attention_mask: Optional attention mask.
            depth_ids: Optional depth bin IDs.
            temperature: Sampling temperature (1.0 = standard, lower = more greedy).

        Returns:
            Tuple of (sampled_actions, probabilities).
            sampled_actions: (batch,) or scalar if input was 1D.
            probabilities: (batch, num_actions) or (num_actions,) if input was 1D.
        """
        # Handle single sample (no batch dim)
        single_sample = input_ids.dim() == 1
        if single_sample:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            if depth_ids is not None:
                depth_ids = depth_ids.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask, depth_ids)
            logits = result['logits']

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                # Greedy
                actions = logits.argmax(dim=-1)
                probs = F.softmax(logits, dim=-1)

        if single_sample:
            actions = actions.squeeze(0)
            probs = probs.squeeze(0)

        return actions, probs

    def predict_action_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        depth_ids: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Get action probabilities as a dictionary.

        Args:
            input_ids: (seq_len,) token IDs for a single sample.
            attention_mask: Optional attention mask.
            depth_ids: Optional depth bin IDs.

        Returns:
            Dict mapping action names to probabilities.
        """
        _, probs = self.sample_action(
            input_ids, attention_mask, depth_ids, temperature=1.0
        )
        probs = probs.cpu().numpy()

        return {
            name: float(probs[i])
            for i, name in enumerate(self.ACTION_NAMES)
        }

    def save_pretrained(self, save_path: str, save_encoder: bool = True) -> None:
        """Save model to disk.

        Args:
            save_path: Directory to save to.
            save_encoder: If True, save full model including encoder.
                         If False, save only actor head.
        """
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save actor head state dict
        actor_path = os.path.join(save_path, 'actor_head.pt')
        torch.save(self.actor_head.state_dict(), actor_path)

        # Save full model if requested
        if save_encoder:
            # Save encoder using transformers
            self.encoder.save_pretrained(save_path)

            # Save actor head separately (will be loaded by our from_pretrained)
            # Also save a copy for transformers-style loading
            import safetensors.torch as st
            st.save_model(self.actor_head, os.path.join(save_path, 'actor_head.safetensors'))

            # Save config with DPO metadata
            config_path = os.path.join(save_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            config['dpo_config'] = {
                'hidden_size': self.hidden_size,
                'num_actions': self.num_actions,
                'actor_hidden': self.actor_head[0].out_features,
            }

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        encoder_path: str | None = None,
        **kwargs,
    ) -> 'DPODoomPolicy':
        """Load model from disk.

        Args:
            model_path: Path to saved DPO model directory.
            encoder_path: Optional separate encoder path (if not in model_path).
            **kwargs: Additional args passed to constructor.

        Returns:
            Loaded DPODoomPolicy instance.
        """
        import os
        import json

        # Determine encoder path
        if encoder_path is None:
            encoder_path = model_path

        # Load config if available
        config_path = os.path.join(model_path, 'config.json')
        dpo_config = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                full_config = json.load(f)
                dpo_config = full_config.get('dpo_config', {})

        # Override with kwargs
        for key in ['hidden_size', 'actor_hidden', 'num_actions']:
            if key in kwargs:
                dpo_config[key] = kwargs[key]

        # Create model
        model = cls(
            encoder_path=encoder_path,
            **dpo_config,
        )

        # Load actor head weights
        actor_path = os.path.join(model_path, 'actor_head.pt')
        if os.path.exists(actor_path):
            model.actor_head.load_state_dict(torch.load(actor_path, map_location='cpu'))
        else:
            # Try safetensors
            import safetensors.torch as st
            st_path = os.path.join(model_path, 'actor_head.safetensors')
            if os.path.exists(st_path):
                state_dict = st.load_file(st_path)
                model.actor_head.load_state_dict(state_dict)

        return model

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and frozen parameters.

        Returns:
            Dict with 'total', 'trainable', and 'frozen' counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        # Break down
        encoder_total = sum(p.numel() for p in self.encoder.parameters())
        actor_total = sum(p.numel() for p in self.actor_head.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'encoder': encoder_total,
            'actor_head': actor_total,
        }
