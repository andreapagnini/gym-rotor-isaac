"""gym-rotor IsaacLab migration — Phase 1 (monolithic, non-equivariant)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Quadcopter-Direct-v0-GymRotor",
    entry_point=f"{__name__}.gymrotor_env:GymRotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gymrotor_env:GymRotorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GymRotorPPORunnerCfg",
    },
)
