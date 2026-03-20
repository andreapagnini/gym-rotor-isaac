"""GymRotor IsaacLab DirectRLEnv — Phase 1 migration.

Implements gym-rotor CoupledWrapper (MONO) dynamics as batched PyTorch Euler
integration inside a DirectRLEnv shell.  No Isaac articulation body is used.

State:  _state (num_envs, 18) = [x(3), v(3), R_vec(9), W(3)]
        R_vec is column-major:  col0=b1, col1=b2, col2=b3  (Fortran reshape order)
Action: (num_envs, 4) in [-1, 1]
        action[:,0] → f_total (total thrust)
        action[:,1:4] → M (body moments, Nm)
Obs:    (num_envs, 23)  (ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm)
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class GymRotorEnvCfg(DirectRLEnvCfg):
    # --- env bookkeeping ---
    episode_length_s: float = 20.0          # 4000 steps @ 200 Hz
    decimation: int = 1                     # one physics call per env step
    action_space: int = 4
    observation_space: int = 23
    state_space: int = 0
    debug_vis: bool = False

    # --- simulation (Isaac time-step; used for step_dt accounting only) ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,                     # 200 Hz, matches gym-rotor
        render_interval=decimation,
    )

    # --- flat terrain (no robot asset) ---
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # --- UAV physical parameters (gym-rotor nominal) ---
    m: float = 2.15          # mass [kg]
    g: float = 9.81          # gravity [m/s^2]
    J1: float = 0.022        # Jxx = Jyy [kg m^2]
    J3: float = 0.035        # Jzz [kg m^2]
    c_tw: float = 2.2        # thrust-to-weight coefficient
    freq: float = 200.0      # simulation frequency [Hz]

    # derived motor limits (same logic as QuadEnv)
    # hover_force = m*g/4,  min_force=0.5,  max_force = c_tw * hover_force
    # avrg_act = (min + max)/2,  scale_act = max - avrg_act

    # --- state limits ---
    x_lim: float = 1.0       # [m]
    v_lim: float = 4.0       # [m/s]
    W_lim: float = 2 * math.pi  # [rad/s]

    # --- integral term parameters ---
    alpha: float = 0.01      # position integral leak
    beta: float = 0.05       # heading integral leak
    eIx_lim: float = 3.0
    eIb1_lim: float = 3.0
    sat_sigma: float = 1.0

    # --- reward coefficients (from args_parse defaults) ---
    Cx: float = 6.0
    CIx: float = 0.1
    Cv: float = 0.4
    Cb1: float = 6.0
    CIb1: float = 0.1
    CW: float = 0.6


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GymRotorEnv(DirectRLEnv):
    cfg: GymRotorEnvCfg

    def __init__(self, cfg: GymRotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- derived motor constants ---
        hover_force = self.cfg.m * self.cfg.g / 4.0
        min_force = 0.5
        max_force = self.cfg.c_tw * hover_force
        self._avrg_act = (min_force + max_force) / 2.0
        self._scale_act = max_force - self._avrg_act
        self._f_min = 4.0 * min_force
        self._f_max = 4.0 * max_force

        # --- inertia tensors (diagonal) ---
        self._J = torch.tensor(
            [self.cfg.J1, self.cfg.J1, self.cfg.J3],
            device=self.device, dtype=torch.float32
        )
        self._J_inv = 1.0 / self._J

        # --- simulation time-step ---
        self._dt = 1.0 / self.cfg.freq

        # --- state tensor: (N, 18) ---
        self._state = torch.zeros(self.num_envs, 18, device=self.device)

        # --- action buffer ---
        self._f = torch.zeros(self.num_envs, device=self.device)   # total thrust [N]
        self._M = torch.zeros(self.num_envs, 3, device=self.device)  # body moments [Nm]

        # --- integral terms: trapezoidal integration ---
        self._eIx = torch.zeros(self.num_envs, 3, device=self.device)
        self._eIb1 = torch.zeros(self.num_envs, device=self.device)
        self._eIx_integrand = torch.zeros(self.num_envs, 3, device=self.device)
        self._eIb1_integrand = torch.zeros(self.num_envs, device=self.device)

        # --- cached normalized errors (computed once per step in _get_dones) ---
        self._ex_norm = torch.zeros(self.num_envs, 3, device=self.device)
        self._eIx_norm = torch.zeros(self.num_envs, 3, device=self.device)
        self._ev_norm = torch.zeros(self.num_envs, 3, device=self.device)
        self._R_vec_obs = torch.zeros(self.num_envs, 9, device=self.device)
        self._eb1_norm = torch.zeros(self.num_envs, device=self.device)
        self._eIb1_norm = torch.zeros(self.num_envs, device=self.device)
        self._eW_norm = torch.zeros(self.num_envs, 3, device=self.device)

        # initialize R to identity for all envs
        I_flat = torch.eye(3, device=self.device).t().reshape(9)  # column-major I
        self._state[:, 6:15] = I_flat.unsqueeze(0).expand(self.num_envs, -1)

    # -----------------------------------------------------------------------
    # Scene setup (terrain + light only; no Isaac rigid body)
    # -----------------------------------------------------------------------

    def _setup_scene(self):
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone environments (empty per-env prim tree — no articulation)
        self.scene.clone_environments(copy_from_source=False)
        # dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -----------------------------------------------------------------------
    # Action processing
    # -----------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Decode normalized actions → total thrust and body moments."""
        a = actions.clamp(-1.0, 1.0)
        # f_total = 4*(scale_act * a0 + avrg_act), clamped to motor limits
        self._f = (4.0 * (self._scale_act * a[:, 0] + self._avrg_act)).clamp(
            self._f_min, self._f_max
        )
        # moments passed directly (units: Nm, no additional scaling)
        self._M = a[:, 1:4].clone()

    def _apply_action(self):
        """Euler-integrate EoM one step in place on self._state."""
        s = self._state
        x = s[:, 0:3]
        v = s[:, 3:6]
        # R_vec stored column-major: col0=b1, col1=b2, col2=b3
        R = self._R_from_state(s)   # (N, 3, 3)
        W = s[:, 15:18]

        b3 = R[:, :, 2]   # thrust direction = R@e3, shape (N, 3)
        f = self._f        # (N,)
        M = self._M        # (N, 3)

        # Equations of motion
        x_dot = v                                                        # (N, 3)
        v_dot = (self.cfg.g * self._e3(x) - (f / self.cfg.m).unsqueeze(1) * b3)  # (N, 3)
        hat_W = self._hat_batch(W)                                       # (N, 3, 3)
        R_dot = torch.bmm(R, hat_W)                                      # (N, 3, 3)
        J_W = self._J.unsqueeze(0) * W                                   # (N, 3)
        hat_W_J_W = torch.bmm(hat_W, J_W.unsqueeze(2)).squeeze(2)        # (N, 3)
        W_dot = self._J_inv.unsqueeze(0) * (-hat_W_J_W + M)              # (N, 3)

        # Euler step
        x_new = x + x_dot * self._dt
        v_new = v + v_dot * self._dt
        R_new = self._reortho(R + R_dot * self._dt)
        W_new = W + W_dot * self._dt

        # Pack back into state
        s[:, 0:3] = x_new
        s[:, 3:6] = v_new
        s[:, 6:15] = self._R_to_state(R_new)
        s[:, 15:18] = W_new

    # -----------------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute and cache normalized errors (updates integrals once per step)
        self._update_norm_errors()

        terminated = (
            (self._ex_norm.abs() >= 1.0).any(dim=1)
            | (self._ev_norm.abs() >= 1.0).any(dim=1)
            | (self._eW_norm.abs() >= 1.0).any(dim=1)
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ex = self._ex_norm
        eIx = self._eIx_norm
        ev = self._ev_norm
        eb1 = self._eb1_norm
        eIb1 = self._eIb1_norm
        eW = self._eW_norm

        raw = (
            -self.cfg.Cx  * (ex  * ex ).sum(dim=1)
            - self.cfg.CIx * (eIx * eIx).sum(dim=1)
            - self.cfg.Cv  * (ev  * ev ).sum(dim=1)
            - self.cfg.Cb1 * eb1.abs()
            - self.cfg.CIb1 * eIb1.pow(2)
            - self.cfg.CW  * (eW  * eW ).sum(dim=1)
        )
        # Change 1: normalize raw reward from [-14, 0] to [0, 1]  (matches source quad.py line 158)
        reward = (raw + 14.0) / 14.0
        # Change 2: crash penalty — terminated envs get -1.0  (matches source quad.py line 162-163)
        reward = torch.where(self.reset_terminated, torch.full_like(reward, -1.0), reward)
        return reward

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._ex_norm,           # 3
                self._eIx_norm,          # 3
                self._ev_norm,           # 3
                self._R_vec_obs,         # 9  (column-major rotation matrix)
                self._eb1_norm.unsqueeze(1),   # 1
                self._eIb1_norm.unsqueeze(1),  # 1
                self._eW_norm,           # 3
            ],
            dim=-1,
        )  # (N, 23)
        return {"policy": obs}

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        n = len(env_ids)
        device = self.device

        # 20% of resets: zero-state (origin + small perturbation)
        # 80%: larger random initial errors
        coin = torch.rand(n, device=device)
        easy = coin < 0.2

        # Position
        x_range = torch.where(easy, torch.full((n,), 0.0, device=device),
                               torch.full((n,), 0.6, device=device))
        x_init = (torch.rand(n, 3, device=device) * 2 - 1) * x_range.unsqueeze(1)

        # Velocity
        v_max = self.cfg.v_lim * 0.5
        v_range = torch.where(easy, torch.zeros(n, device=device),
                               torch.full((n,), v_max, device=device))
        v_init = (torch.rand(n, 3, device=device) * 2 - 1) * v_range.unsqueeze(1)

        # Angular velocity
        W_max = self.cfg.W_lim * 0.5
        W_range = torch.where(easy, torch.zeros(n, device=device),
                               torch.full((n,), W_max, device=device))
        W_init = (torch.rand(n, 3, device=device) * 2 - 1) * W_range.unsqueeze(1)

        # Rotation: random roll-pitch (±50 deg), random yaw in [-π, π]
        R_deg = 50.0 * math.pi / 180.0
        rp_range = torch.where(easy, torch.zeros(n, device=device),
                                torch.full((n,), R_deg, device=device))
        roll  = (torch.rand(n, device=device) * 2 - 1) * rp_range
        pitch = (torch.rand(n, device=device) * 2 - 1) * rp_range
        yaw   = (torch.rand(n, device=device) * 2 - 1) * math.pi

        R_init = self._euler_to_R(roll, pitch, yaw)  # (n, 3, 3)
        R_init = self._reortho(R_init)

        # Write into state
        self._state[env_ids, 0:3]  = x_init
        self._state[env_ids, 3:6]  = v_init
        self._state[env_ids, 6:15] = self._R_to_state(R_init)
        self._state[env_ids, 15:18] = W_init

        # Reset integral terms for these envs
        self._eIx[env_ids]           = 0.0
        self._eIb1[env_ids]          = 0.0
        self._eIx_integrand[env_ids] = 0.0
        self._eIb1_integrand[env_ids] = 0.0

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _R_from_state(self, s: torch.Tensor) -> torch.Tensor:
        """Reconstruct (N,3,3) rotation matrix from column-major R_vec in state."""
        # s[:, 6:9]=col0, s[:, 9:12]=col1, s[:, 12:15]=col2
        return torch.stack([s[:, 6:9], s[:, 9:12], s[:, 12:15]], dim=-1)  # (N,3,3)

    def _R_to_state(self, R: torch.Tensor) -> torch.Tensor:
        """Pack (N,3,3) rotation matrix back to column-major 9-vector."""
        # col0, col1, col2 → cat along last dim
        return torch.cat([R[:, :, 0], R[:, :, 1], R[:, :, 2]], dim=1)  # (N,9)

    @staticmethod
    def _hat_batch(W: torch.Tensor) -> torch.Tensor:
        """Batched so(3) hat map. W: (N,3) → (N,3,3)."""
        N = W.shape[0]
        H = torch.zeros(N, 3, 3, device=W.device, dtype=W.dtype)
        H[:, 0, 1] = -W[:, 2]
        H[:, 0, 2] =  W[:, 1]
        H[:, 1, 0] =  W[:, 2]
        H[:, 1, 2] = -W[:, 0]
        H[:, 2, 0] = -W[:, 1]
        H[:, 2, 1] =  W[:, 0]
        return H

    @staticmethod
    def _reortho(R: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt re-orthonormalization of columns. R: (N,3,3)."""
        c0 = R[:, :, 0]
        c1 = R[:, :, 1]
        eps = 1e-8
        c0 = c0 / c0.norm(dim=1, keepdim=True).clamp(min=eps)
        c1 = c1 - (c1 * c0).sum(dim=1, keepdim=True) * c0
        c1 = c1 / c1.norm(dim=1, keepdim=True).clamp(min=eps)
        c2 = torch.linalg.cross(c0, c1)
        return torch.stack([c0, c1, c2], dim=-1)

    def _e3(self, ref: torch.Tensor) -> torch.Tensor:
        """Return e3 = [0,0,1] broadcast to (N,3)."""
        e3 = torch.zeros_like(ref)
        e3[:, 2] = 1.0
        return e3

    @staticmethod
    def _euler_to_R(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """XYZ Euler angles → rotation matrix. Inputs: (N,) radians → output (N,3,3)."""
        cr, sr = torch.cos(roll),  torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw),   torch.sin(yaw)
        N = roll.shape[0]
        R = torch.zeros(N, 3, 3, device=roll.device, dtype=roll.dtype)
        # R = Rz @ Ry @ Rx  (extrinsic XYZ = intrinsic ZYX)
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        return R

    def _update_norm_errors(self):
        """Compute and cache normalized error state, updating integral terms."""
        s = self._state
        x = s[:, 0:3]
        v = s[:, 3:6]
        W = s[:, 15:18]
        R = self._R_from_state(s)   # (N, 3, 3)

        ex_norm = x / self.cfg.x_lim
        ev_norm = v / self.cfg.v_lim
        eW_norm = W / self.cfg.W_lim
        R_vec   = s[:, 6:15]        # column-major, shape (N, 9)

        # Heading error eb1 (scalar, normalized to [-1, 1] via /pi)
        b3 = R[:, :, 2]   # (N, 3)
        b1 = R[:, :, 0]   # (N, 3)
        b2 = R[:, :, 1]   # (N, 3)
        b1d = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        b1d_dot_b3 = (b1d.unsqueeze(0) * b3).sum(dim=1, keepdim=True)  # (N, 1)
        b1c = b1d.unsqueeze(0) - b1d_dot_b3 * b3                       # (N, 3)
        eb1 = torch.atan2(
            -(b1c * b2).sum(dim=1),
             (b1c * b1).sum(dim=1)
        )  # (N,)
        eb1_norm = eb1 / math.pi

        # Integral terms — trapezoidal rule
        integrand_x = -self.cfg.alpha * self._eIx + ex_norm * self.cfg.x_lim
        self._eIx = self._eIx + (self._eIx_integrand + integrand_x) * (self._dt / 2.0)
        self._eIx_integrand = integrand_x
        eIx_norm = self._eIx.div(self.cfg.eIx_lim).clamp(-self.cfg.sat_sigma, self.cfg.sat_sigma)

        integrand_b1 = -self.cfg.beta * self._eIb1 + eb1_norm * math.pi
        self._eIb1 = self._eIb1 + (self._eIb1_integrand + integrand_b1) * (self._dt / 2.0)
        self._eIb1_integrand = integrand_b1
        eIb1_norm = self._eIb1.div(self.cfg.eIb1_lim).clamp(-self.cfg.sat_sigma, self.cfg.sat_sigma)

        # Cache
        self._ex_norm    = ex_norm
        self._eIx_norm   = eIx_norm
        self._ev_norm    = ev_norm
        self._R_vec_obs  = R_vec
        self._eb1_norm   = eb1_norm
        self._eIb1_norm  = eIb1_norm
        self._eW_norm    = eW_norm
