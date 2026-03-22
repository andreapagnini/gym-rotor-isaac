"""Test registration of both Phase 1 (MLP) and Phase 2 (EMLP) tasks."""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gym_rotor_isaac
import gymnasium as gym

print("\n=== Task Registration Test ===\n")

for task_id in [
    "Isaac-Quadcopter-Direct-v0-GymRotor",
    "Isaac-Quadcopter-Direct-v0-GymRotor-EMLP",
]:
    spec = gym.spec(task_id)
    print(f"✓ OK: {spec.id}")
    print(f"  rsl_rl_cfg: {spec.kwargs['rsl_rl_cfg_entry_point']}")
    print()

print("=== All tasks registered successfully ===\n")

simulation_app.close()
