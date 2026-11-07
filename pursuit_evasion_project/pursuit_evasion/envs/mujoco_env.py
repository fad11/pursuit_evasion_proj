import mujoco
import numpy as np
from .base_env import BasePursuitEvasionEnv
from ._assets import resolve_asset

class MujocoPursuitEvasionEnv(BasePursuitEvasionEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.use_physics_backend, self.mass = True, 1.0
        self.model = mujoco.MjModel.from_xml_path(resolve_asset("world.xml"))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt
        self._set_gravity(self.gravity)
        self.viewer, self.renderer = None, None
        self.p_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pursuer")
        self.e_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "evader")
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs); mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:self.dim] = self.p_pos
        self.data.qpos[7:7 + self.dim] = self.e_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]; self.data.qpos[10:14] = [1, 0, 0, 0]; mujoco.mj_forward(self.model, self.data)
        return obs, info
    def _step_physics(self, a_p, a_e):
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.p_id, 0:self.dim] = a_p * self.mass
        self.data.xfrc_applied[self.e_id, 0:self.dim] = a_e * self.mass
        mujoco.mj_step(self.model, self.data)
        self.p_pos = self.data.qpos[0:self.dim].copy()
        self.p_vel = self.data.qvel[0:self.dim].copy()
        self.e_pos = self.data.qpos[7:7 + self.dim].copy()
        self.e_vel = self.data.qvel[6:6 + self.dim].copy()

    def _set_gravity(self, gravity):
        if hasattr(self, "model"):
            self.model.opt.gravity[2] = -gravity

    def _sync_physics_state(self):
        self.data.qpos[0:self.dim] = self.p_pos
        self.data.qvel[0:self.dim] = self.p_vel
        self.data.qpos[7:7 + self.dim] = self.e_pos
        self.data.qvel[6:6 + self.dim] = self.e_vel
        mujoco.mj_forward(self.model, self.data)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer; self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.renderer is None: self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="top_down"); return self.renderer.render()
    def close(self):
        if self.viewer: self.viewer.close()
        if self.renderer: self.renderer.close()
