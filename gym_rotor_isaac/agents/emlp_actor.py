"""
EMLPActor: Wrapper for EMLP_MONO_Actor_PPO to work with rsl_rl's MLPModel interface.

This adapter allows the equivariant EMLP actor network from the source gym-rotor repo
to be used as a drop-in replacement for rsl_rl's MLPModel.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

# Import EMLP components from the vendored library
import sys
from pathlib import Path
# Add parent directory to path to import ppo_emlp
source_path = Path(__file__).parent.parent.parent.parent / "eqRL_quadrotor" / "gym-rotor-main"
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))

from algos.ppo.ppo_emlp import EMLP_MONO_Actor_PPO


class EMLPActor(nn.Module):
    """Equivariant MLP actor wrapper compatible with rsl_rl PPO."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,              # "actor"
        output_dim: int,           # num_actions = 4
        hidden_dims: list[int] | None = None,  # [16] default
        hidden_num: int = 2,       # Number of hidden layers
        obs_normalization: bool = False,  # Ignored (inputs already normalized)
        distribution_cfg: dict | None = None,  # Ignored (EMLP manages its own distribution)
        **kwargs,
    ):
        """Initialize EMLPActor wrapper.

        Args:
            obs: Observation TensorDict from environment
            obs_groups: Dict mapping obs sets to lists of obs group names
            obs_set: Which obs set this model uses ("actor")
            output_dim: Action dimension (4 for MONO)
            hidden_dims: Hidden layer dimensions (list, will extract [0])
            hidden_num: Number of EMLP hidden layers
            obs_normalization: Ignored (EMLP inputs already in [-1,1])
            distribution_cfg: Ignored (EMLP has built-in Normal distribution)
            **kwargs: Additional args (ignored)
        """
        super().__init__()

        # Extract device from obs
        self.device = obs["policy"].device

        # Extract hidden_dim from hidden_dims list (rsl_rl passes a list)
        if hidden_dims is None:
            hidden_dim = 16  # Default from args_parse.py
        else:
            hidden_dim = hidden_dims[0] if isinstance(hidden_dims, list) else hidden_dims

        # Synthesize args namespace for EMLP constructor
        args = self._make_args(output_dim, hidden_dim)

        # Build EMLP actor (groups created on self.device)
        self._emlp = EMLP_MONO_Actor_PPO(
            args,
            agent_id=0,
            hidden_num=hidden_num,
            log_std=kwargs.get('init_log_std', 0.0)
        )

        # Distribution storage (populated by forward())
        self._distribution: Normal | None = None

        # Store obs_groups for TensorDict extraction
        self.obs_groups = obs_groups[obs_set]  # ["policy"]
        self.obs_dim = sum(obs[g].shape[-1] for g in self.obs_groups)

    def _make_args(self, num_actions: int, hidden_dim: int):
        """Synthesize args namespace for EMLP constructor."""
        class Args:
            pass
        args = Args()
        args.device = self.device
        args.action_dim_n = [num_actions]      # [4] for MONO
        args.actor_hidden_dim = [hidden_dim]   # [16] default
        return args

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False
    ) -> torch.Tensor:
        """Forward pass through EMLP actor.

        Args:
            obs: Observation TensorDict with "policy" key
            masks: Optional masks (unused, for recurrent models)
            hidden_state: Optional hidden state (unused, non-recurrent)
            stochastic_output: If True, sample from distribution; else return mean

        Returns:
            Action tensor (sampled if stochastic_output=True, else mean)
        """
        # Extract flat tensor from TensorDict
        obs_flat = torch.cat([obs[g] for g in self.obs_groups], dim=-1)

        # Get distribution from EMLP
        self._distribution = self._emlp.get_dist(obs_flat)

        if stochastic_output:
            return self._distribution.sample()
        else:
            return self._distribution.mean  # deterministic = mean

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current distribution."""
        return (self._distribution.mean, self._distribution.stddev)

    @property
    def output_std(self) -> torch.Tensor:
        """Return standard deviation (for logging)."""
        return self._distribution.stddev[0]

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy of the current distribution."""
        return self._distribution.entropy().sum(dim=-1)

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        return self._distribution.log_prob(actions).sum(dim=-1)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence KL(old || new) between two Gaussian distributions."""
        old_dist = Normal(old_params[0], old_params[1])
        new_dist = Normal(new_params[0], new_params[1])
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

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
