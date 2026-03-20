"""Minimal registration check — no env instantiation, no GPU/scene allocation."""
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gym_rotor_isaac  # noqa: F401 — triggers gym.register(...)
import gymnasium as gym

OUT = "/tmp/gymrotor_reg_check.txt"
try:
    env_spec = gym.spec("Isaac-Quadcopter-Direct-v0-GymRotor")
    msg = (
        f"Registration OK: {env_spec.id}\n"
        f"  entry_point : {env_spec.entry_point}\n"
        f"  kwargs      : {env_spec.kwargs}\n"
    )
    result = "PASS"
except Exception as e:
    msg = f"Registration FAILED: {e}\n"
    result = "FAIL"

# write to file (bypasses any stdout buffering)
with open(OUT, "w") as f:
    f.write(msg)

# also print to stdout with flush
print(msg, flush=True)
sys.stdout.flush()

simulation_app.close()

# re-print after close so it definitely appears in terminal
print(f"[test_registration] result={result}", flush=True)
