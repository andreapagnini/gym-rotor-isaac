"""
EMLPCritic: Wrapper for EMLP_MONO_Critic_PPO to work with rsl_rl's MLPModel interface.

This adapter allows the equivariant EMLP critic network from the source gym-rotor repo
to be used as a drop-in replacement for rsl_rl's MLPModel.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict

# Import EMLP components from the vendored library
import sys
from pathlib import Path
# Add parent directory to path to import ppo_emlp
source_path = Path(__file__).parent.parent.parent.parent / "eqRL_quadrotor" / "gym-rotor-main"
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

from algos.ppo.ppo_emlp import EMLP_MONO_Critic_PPO


class EMLPCritic(nn.Module):
    """Equivariant MLP critic wrapper compatible with rsl_rl PPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,              # "critic"
        output_dim: int,           # 1 (value function scalar)
        hidden_dims: list[int] | None = None,  # [62] default
        hidden_num: int = 2,       # Number of hidden layers
        obs_normalization: bool = False,  # Ignored (inputs already normalized)
        distribution_cfg: dict | None = None,  # Ignored (critic has no distribution)
        **kwargs,
    ):
        """Initialize EMLPCritic wrapper.

        Args:
            obs: Observation TensorDict from environment
            obs_groups: Dict mapping obs sets to lists of obs group names
            obs_set: Which obs set this model uses ("critic")
            output_dim: Output dimension (1 for value function)
            hidden_dims: Hidden layer dimensions (list, will extract [0])
            hidden_num: Number of EMLP hidden layers
            obs_normalization: Ignored (EMLP inputs already in [-1,1])
            distribution_cfg: Ignored (critic doesn't use distribution)
            **kwargs: Additional args (ignored)
        """
        super().__init__()

        # Extract device from obs
        self.device = obs["policy"].device

        # Extract hidden_dim from hidden_dims list (rsl_rl passes a list)
        if hidden_dims is None:
            hidden_dim = 62  # Default from args_parse.py
        else:
            hidden_dim = hidden_dims[0] if isinstance(hidden_dims, list) else hidden_dims

        # Synthesize args namespace for EMLP constructor
        args = self._make_args(hidden_dim)

        # Build EMLP critic (groups created on self.device)
        self._emlp = EMLP_MONO_Critic_PPO(
            args,
            agent_id=0,
            hidden_num=hidden_num
        )

        # Store obs_groups for TensorDict extraction
        self.obs_groups = obs_groups[obs_set]  # ["policy"]
        self.obs_dim = sum(obs[g].shape[-1] for g in self.obs_groups)

    def _make_args(self, hidden_dim: int):
        """Synthesize args namespace for EMLP constructor."""
        class Args:
            pass
        args = Args()
        args.device = self.device
        args.critic_hidden_dim = hidden_dim  # 62 default (int, not list)
        return args

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        """Forward pass through EMLP critic.

        Args:
            obs: Observation TensorDict with "policy" key
            masks: Optional masks (unused, for recurrent models)
            hidden_state: Optional hidden state (unused, non-recurrent)

        Returns:
            Value tensor (N, 1)
        """
        # Extract flat tensor from TensorDict
        obs_flat = torch.cat([obs[g] for g in self.obs_groups], dim=-1)

        # Get value from EMLP
        return self._emlp(obs_flat)

    # No-ops for non-recurrent models
    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset recurrent state (no-op for non-recurrent)."""
        pass

    def get_hidden_state(self):
        """Return hidden state (None for non-recurrent)."""
        return None

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization (no-op, inputs already normalized)."""
        pass
