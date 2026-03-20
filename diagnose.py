"""Diagnostic script — investigate plateau root cause.
All output written to /tmp/diag_out.txt (stdout is eaten by isaaclab.sh).
"""
import argparse, math, builtins

OUTFILE = "/tmp/diag_out.txt"
_outf = open(OUTFILE, "w", buffering=1)

def p(*args, **kwargs):
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)
    builtins.print(*args, file=_outf, flush=True)

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gym_rotor_isaac  # noqa: F401
from gym_rotor_isaac.gymrotor_env import GymRotorEnvCfg, GymRotorEnv


def print_errors(uw, label):
    ex  = uw._ex_norm[0];  ev = uw._ev_norm[0];  eW = uw._eW_norm[0]
    eb1 = uw._eb1_norm[0].item()
    eIx = uw._eIx_norm[0];  eIb1 = uw._eIb1_norm[0].item()
    done_x = (ex.abs() >= 1.0).any().item()
    done_v = (ev.abs() >= 1.0).any().item()
    done_W = (eW.abs() >= 1.0).any().item()
    p(f"  [{label}]")
    p(f"    ex_norm  = {[f'{v:.4f}' for v in ex.tolist()]}  |max|={ex.abs().max():.4f}  done?{done_x}")
    p(f"    ev_norm  = {[f'{v:.4f}' for v in ev.tolist()]}  |max|={ev.abs().max():.4f}  done?{done_v}")
    p(f"    eW_norm  = {[f'{v:.4f}' for v in eW.tolist()]}  |max|={eW.abs().max():.4f}  done?{done_W}")
    p(f"    eb1_norm = {eb1:.4f}")
    p(f"    eIx_norm = {[f'{v:.4f}' for v in eIx.tolist()]}")
    p(f"    eIb1_norm= {eIb1:.4f}")
    p(f"    done triggered? {done_x or done_v or done_W}")


def do_step(uw, action):
    uw._pre_physics_step(action)
    uw._apply_action()
    uw.scene.write_data_to_sim()
    uw.sim.step()
    uw.scene.update(uw.physics_dt)
    terminated, truncated = uw._get_dones()
    reward = uw._get_rewards()
    uw._get_observations()
    uw.episode_length_buf += 1
    reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
    if len(reset_ids):
        uw._reset_idx(reset_ids)
    return reward, terminated, truncated


# ── create 1-env instance ──────────────────────────────────────────────────
cfg = GymRotorEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 2.5
uw = GymRotorEnv(cfg)
uw.reset()

zero = torch.zeros(1, 4, device=uw.device)

# ═══════════════════════════════════════════════════════════════════════════
p("\n" + "="*60)
p("DIAGNOSTIC 2 — initial state after reset (env 0)")
p("="*60)
s = uw._state[0]
x_s = s[0:3]; v_s = s[3:6]; R_vec = s[6:15]; W_s = s[15:18]
R = torch.stack([R_vec[0:3], R_vec[3:6], R_vec[6:9]], dim=-1)
det_R     = torch.linalg.det(R).item()
ortho_err = (R.T @ R - torch.eye(3, device=R.device)).abs().max().item()
p(f"  x        = {[f'{v:.4f}' for v in x_s.tolist()]}")
p(f"  v        = {[f'{v:.4f}' for v in v_s.tolist()]}")
p(f"  W        = {[f'{v:.4f}' for v in W_s.tolist()]}")
p(f"  R col0   = {[f'{v:.4f}' for v in R_vec[0:3].tolist()]}  (b1)")
p(f"  R col1   = {[f'{v:.4f}' for v in R_vec[3:6].tolist()]}  (b2)")
p(f"  R col2   = {[f'{v:.4f}' for v in R_vec[6:9].tolist()]}  (b3)")
p(f"  det(R)   = {det_R:.6f}  (should be ~1.0)")
p(f"  max|R^TR-I| = {ortho_err:.2e}  (should be ~0)")

# ═══════════════════════════════════════════════════════════════════════════
p("\n" + "="*60)
p("DIAGNOSTIC 1 — f_total and M for steps 1-3  (action = zero)")
p("="*60)
m_val = uw.cfg.m;  g_val = uw.cfg.g
p(f"  avrg_act  = {uw._avrg_act:.4f} N  (per-motor avg)")
p(f"  scale_act = {uw._scale_act:.4f} N  (per-motor scale)")
p(f"  f_min/max = [{uw._f_min:.4f}, {uw._f_max:.4f}] N  (4-motor total)")
p(f"  m*g       = {m_val*g_val:.4f} N  (hover total thrust)")
p(f"  action=0 decodes to f_total = 4*avrg = {4*uw._avrg_act:.4f} N")

uw.reset()
for step in range(1, 4):
    reward, terminated, truncated = do_step(uw, zero)
    f_val = uw._f[0].item()
    M_val = [f'{v:.4f}' for v in uw._M[0].tolist()]
    rew   = reward[0].item()
    term  = terminated[0].item()
    p(f"  step {step}: f_total={f_val:.4f} N   M={M_val}   reward={rew:.4f}   terminated={term}")
    if term:
        p(f"           EPISODE ENDED — resetting")
        uw.reset()

# ═══════════════════════════════════════════════════════════════════════════
p("\n" + "="*60)
p("DIAGNOSTIC 3 — error terms at step 1 and first termination (<=30 steps)")
p("="*60)
uw.reset()
for step in range(1, 31):
    reward, terminated, truncated = do_step(uw, zero)
    term  = terminated[0].item()
    trunc = truncated[0].item()
    if step == 1:
        print_errors(uw, f"step {step}")
    if step == 21 or term or trunc:
        suffix = " TERMINATED" if term else " TRUNCATED" if trunc else ""
        print_errors(uw, f"step {step}{suffix}")
        if term or trunc:
            p(f"  Episode ended at step {step}")
            break

# ═══════════════════════════════════════════════════════════════════════════
p("\n" + "="*60)
p("DIAGNOSTIC 4 — reward normalization")
p("="*60)
uw.reset()
reward, terminated, truncated = do_step(uw, zero)
ret_val   = reward[0].item()
coeff_sum = uw.cfg.Cx + uw.cfg.CIx + uw.cfg.Cv + uw.cfg.Cb1 + uw.cfg.CIb1 + uw.cfg.CW
rmin      = -math.ceil(coeff_sum)
p(f"  reward from do_step()              = {ret_val:.4f}")
p(f"  coefficients: Cx={uw.cfg.Cx} CIx={uw.cfg.CIx} Cv={uw.cfg.Cv} "
  f"Cb1={uw.cfg.Cb1} CIb1={uw.cfg.CIb1} CW={uw.cfg.CW}")
p(f"  sum of coeffs                      = {coeff_sum:.1f}")
p(f"  theoretical reward_min             = {rmin}  (-ceil({coeff_sum:.1f}))")
p(f"  Is reward in [0,1]?                {'YES' if 0.0 <= ret_val <= 1.0 else 'NO — raw negative'}")
if rmin != 0:
    normed = (ret_val - rmin) / (0.0 - rmin)
    p(f"  If normalized via interp:          {normed:.4f}")
p(f"  Source gym-rotor returns:          interp(raw, [{rmin}, 0], [0, 1])")
p(f"  Our implementation returns:        raw value (no normalization)")

uw.close()
simulation_app.close()
p("\nDiagnostics complete.")
_outf.close()
