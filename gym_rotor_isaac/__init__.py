"""gym-rotor IsaacLab migration — Phase 1 (monolithic, non-equivariant) + Phase 2 (equivariant EMLP)."""

import gymnasium as gym

from . import agents

# Phase 1: Monolithic non-equivariant MLP baseline
gym.register(
    id="Isaac-Quadcopter-Direct-v0-GymRotor",
    entry_point=f"{__name__}.gymrotor_env:GymRotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gymrotor_env:GymRotorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GymRotorPPORunnerCfg",
    },
)

# Phase 2: Monolithic equivariant EMLP
gym.register(
    id="Isaac-Quadcopter-Direct-v0-GymRotor-EMLP",
    entry_point=f"{__name__}.gymrotor_env:GymRotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gymrotor_env:GymRotorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GymRotorEMLPPPORunnerCfg",
    },
)
