from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np
import random
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}

class Go1MujocoEnv(MujocoEnv):
    """Optimized bipedal walking environment for Go1 robot."""
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }
    
    BIPEDAL_READY_JOINTS = np.array([
        0.0, 3.8, -2.2,    # FR
        0.0, 3.8, -2.2,    # FL
        0.0, 2.6, -1.4,    # RR
        0.0, 2.6, -1.4,    # RL
    ])
    
    def __init__(self, ctrl_type="torque", biped=False, rand_power=0.0, **kwargs):
        model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml")
        self.biped = biped
        self._rand_power = rand_power
        
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 60,
        }
        
        self._last_render_time = -1.0
        self._max_episode_time_sec = 20.0
        self._step = 0
        self._front_feet_touched = False
        self._episode_count = 0
        self._success_count = 0
        
        self.reward_weights = {
            "linear_vel_tracking": 2.0,
            "angular_vel_tracking": 1.0,
            "healthy": 1.0,
            "feet_airtime": 5.0,
            "biped_front_feet_off_ground": 40.0,
            "biped_perfect_upright": 60.0,
            "forward_velocity": 25.0,
            "balance_stability": 20.0,
        }
        
        self.cost_weights = {
            "torque": 0.0001,
            "vertical_vel": 1.5,
            "xy_angular_vel": 0.03,
            "action_rate": 0.005,
            "joint_limit": 8.0,
            "joint_velocity": 0.005,
            "joint_acceleration": 1.0e-4,
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.05,
        }
        
        if self.biped:
            self.reward_weights["biped_upright"] = 20.0
            self.cost_weights["biped_front_contact"] = 80.0
            self.cost_weights["biped_rear_feet_airborne"] = 3.0
            self.cost_weights["biped_front_foot_height"] = 6.0
            self.cost_weights["biped_crossed_legs"] = 4.0
            self.cost_weights["biped_low_rear_hips"] = 7.0
            self.cost_weights["biped_front_feet_below_hips"] = 80.0
            self.cost_weights["biped_abduction_joints"] = 0.5
            self.cost_weights["biped_unwanted_contact"] = 120.0
            self.cost_weights["self_collision"] = 20.0
            self.cost_weights["biped_body_height"] = 4.0
            self.cost_weights["biped_roll_stability"] = 6.0
            self.cost_weights["biped_pitch_stability"] = 8.0
            
        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])
        
        self._desired_velocity_min = np.array([0.0, -0.1, -0.1])
        self._desired_velocity_max = np.array([0.3, 0.1, 0.1])
        self._desired_velocity = self._sample_desired_vel()
        
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        
        self._tracking_velocity_sigma = 0.25
        
        self._healthy_z_range = (0.20, 1.8)
        self._healthy_pitch_range = (-np.pi, 0.0)
        self._healthy_roll_range = (-np.deg2rad(85), np.deg2rad(85))
        
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]
        self._cfrc_ext_front_feet_indices = [4, 7]
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
        
        dof_position_limit_multiplier = 0.95
        ctrl_range_offset = (
            0.5 * (1 - dof_position_limit_multiplier) * 
            (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        )
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset
        
        self._reset_noise_scale = 0.05
        self._last_action = np.zeros(12)
        self._last_feet_contact_forces = np.zeros(4)
        self._clip_obs_threshold = 100.0
        self._balance_history = []
        self._max_balance_history = 10
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64
        )
        
        feet_site = ["FR", "FL", "RR", "RL"]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }
        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )
        
        if self.biped:
            self._initialize_biped_body_ids()
    
    def _initialize_biped_body_ids(self):
        front_knee_body_names = ["FR_calf", "FL_calf"]
        self._front_knee_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in front_knee_body_names
        ]
        
        self._front_feet_site_ids = [
            self._feet_site_name_to_id["FR"],
            self._feet_site_name_to_id["FL"]
        ]
        
        rear_hip_body_names = ["RR_hip", "RL_hip"]
        self._rear_hip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in rear_hip_body_names
        ]
        self._rear_hips_min_height = 0.18
        
        front_hip_body_names = ["FR_hip", "FL_hip"]
        self._front_hip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in front_hip_body_names
        ]
        
        unwanted_contact_body_names = [
            "trunk", "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh", "FR_calf", "FL_calf",
        ]
        self._unwanted_contact_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in unwanted_contact_body_names
        ]
        
        self._front_right_limb_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in ["FR_hip", "FR_thigh", "FR_calf"]
        }
        self._front_left_limb_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in ["FL_hip", "FL_thigh", "FL_calf"]
        }
        self._rear_right_limb_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in ["RR_hip", "RR_thigh", "RR_calf"]
        }
        self._rear_left_limb_body_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in ["RL_hip", "RL_thigh", "RL_calf"]
        }
    
    @property
    def curriculum_factor(self):
        progress = min(1.0, self._episode_count / 1000.0)
        return progress
    
    @property
    def forward_velocity_reward(self):
        forward_vel = self.data.qvel[0]
        target_vel = 0.2
        vel_error = abs(forward_vel - target_vel)
        reward = np.exp(-5.0 * vel_error)
        #if forward_vel < 0:
        #    reward *= 0.1
        return reward
    
    @property
    def balance_stability_reward(self):
        w, x, y, z = self.data.qpos[3:7]
        roll, pitch, _ = self.euler_from_quaternion(w, x, y, z)
        target_pitch = np.deg2rad(-90)
        pitch_error = abs(pitch - target_pitch)
        roll_error = abs(roll)
        balance_score = np.exp(-3.0 * (pitch_error + roll_error))
        self._balance_history.append(balance_score)
        if len(self._balance_history) > self._max_balance_history:
            self._balance_history.pop(0)
        return np.mean(self._balance_history) if self._balance_history else balance_score
    
    @property
    def adaptive_joint_acceleration_cost(self):
        joint_velocities = np.abs(self.data.qvel[6:])
        joint_accelerations = self.data.qacc[6:]
        epsilon = 1e-6
        penalty_scale = 1.0 + self.curriculum_factor * 2.0
        dynamic_penalty = np.sum(
            np.square(joint_accelerations) / (joint_velocities + epsilon)
        ) * penalty_scale
        return dynamic_penalty
    
    @property
    def biped_perfect_upright_reward(self):
        w, x, y, z = self.data.qpos[3:7]
        _, pitch, _ = self.euler_from_quaternion(w, x, y, z)
        target_pitch = np.deg2rad(-90)
        pitch_error = abs(pitch - target_pitch)
        max_error = np.deg2rad(60)
        if pitch_error > max_error:
            return 0.0
        reward = np.exp(-8 * pitch_error / max_error)
        if pitch_error < np.deg2rad(10):
            reward *= 1.5
        return reward
    
    @property
    def biped_front_feet_off_ground_reward(self):
        front_contact_forces = self.front_feet_contact_forces
        both_feet_off_ground = np.all(front_contact_forces <= 1.0)
        return float(both_feet_off_ground)
    
    @property
    def acceleration_cost(self):
        joint_velocities = np.abs(self.data.qvel[6:])
        joint_accelerations = self.data.qacc[6:]
        epsilon = 1e-6
        dynamic_penalty = np.sum(
            np.square(joint_accelerations) / (joint_velocities + epsilon)
        )
        return dynamic_penalty
    
    @property
    def self_collision_cost(self):
        cost = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            
            if (body1_id in self._front_right_limb_body_ids and 
                body2_id in self._front_left_limb_body_ids):
                cost += 0.5
            elif (body2_id in self._front_right_limb_body_ids and 
                  body1_id in self._front_left_limb_body_ids):
                cost += 0.5
            elif (body1_id in self._rear_right_limb_body_ids and 
                  body2_id in self._rear_left_limb_body_ids):
                cost += 0.5
            elif (body2_id in self._rear_right_limb_body_ids and 
                  body1_id in self._rear_left_limb_body_ids):
                cost += 0.5
        return cost
    
    @property
    def biped_crossed_legs_cost(self):
        rear_hips_pos = self.data.xpos[self._rear_hip_body_ids]
        y_rr = rear_hips_pos[0, 1]
        y_rl = rear_hips_pos[1, 1]
        cost = max(0, y_rr - y_rl)
        return cost
    
    @property
    def biped_low_rear_hips_cost(self):
        rear_hips_pos = self.data.xpos[self._rear_hip_body_ids]
        hips_z = rear_hips_pos[:, 2]
        height_difference = self._rear_hips_min_height - hips_z
        cost = np.sum(height_difference.clip(min=0.0))
        return cost * 8.0
    
    @property
    def biped_front_feet_below_hips_cost(self):
        front_feet_pos = self.data.site_xpos[self._front_feet_site_ids]
        front_hips_pos = self.data.xpos[self._front_hip_body_ids]
        feet_z = front_feet_pos[:, 2]
        hips_z = front_hips_pos[:, 2]
        height_difference = hips_z - feet_z
        cost = np.sum(np.square(height_difference.clip(min=0.0)))
        return cost
    
    @property
    def trunk_forward_axis_in_world(self):
        return self.data.xmat[self._main_body_id].reshape(3, 3)[:, 0]
    
    @property
    def front_feet_contact_forces(self):
        front_feet_forces = self.data.cfrc_ext[self._cfrc_ext_front_feet_indices]
        return np.linalg.norm(front_feet_forces, axis=1)
    
    @property
    def biped_upright_reward(self):
        world_up_vector = np.array([0, 0, 1])
        trunk_forward_vector = self.trunk_forward_axis_in_world
        alignment = np.dot(trunk_forward_vector, world_up_vector)
        return max(0, alignment)
    
    @property
    def biped_front_foot_height_cost(self):
        front_feet_pos = self.data.site_xpos[self._front_feet_site_ids]
        front_knees_pos = self.data.xpos[self._front_knee_body_ids]
        feet_z = front_feet_pos[:, 2]
        knees_z = front_knees_pos[:, 2]
        height_difference = knees_z - feet_z
        cost = np.sum(height_difference.clip(min=0.0))
        return cost
    
    @property
    def biped_body_height_cost(self):
        z_pos = self.data.qpos[2]
        if z_pos < 0.25:
            penalty = np.exp((0.25 - z_pos) * 10) - 1
            return penalty
        return 0.0
    
    @property
    def biped_roll_stability_cost(self):
        w, x, y, z = self.data.qpos[3:7]
        roll, _, _ = self.euler_from_quaternion(w, x, y, z)
        soft_limit = np.deg2rad(35)
        hard_limit = np.deg2rad(50)
        roll_abs = abs(roll)
        if roll_abs > hard_limit:
            return np.square((roll_abs - soft_limit) * 4)
        elif roll_abs > soft_limit:
            return (roll_abs - soft_limit) * 1.5
        return 0.0
    
    @property
    def biped_pitch_stability_cost(self):
        w, x, y, z = self.data.qpos[3:7]
        _, pitch, _ = self.euler_from_quaternion(w, x, y, z)
        target_pitch = np.deg2rad(-90)
        pitch_error = abs(pitch - target_pitch)
        soft_limit = np.deg2rad(35)
        hard_limit = np.deg2rad(65)
        if pitch_error > hard_limit:
            return np.square((pitch_error - soft_limit) * 2.5)
        elif pitch_error > soft_limit:
            return (pitch_error - soft_limit) * 1.2
        return 0.0
    
    @property
    def biped_front_contact_cost(self):
        contact_forces = self.front_feet_contact_forces
        return np.sum(np.power(contact_forces, 1.5))
    
    @property
    def biped_abduction_joints_cost(self):
        abduction_joints_indices = [0, 3, 6, 9]
        dofs_position = self.data.qpos[7:]
        abduction_angles = dofs_position[abduction_joints_indices]
        return np.sum(np.square(abduction_angles))
    
    @property
    def biped_unwanted_contact_cost(self):
        contact_forces = self.data.cfrc_ext[self._unwanted_contact_body_ids]
        cost = np.sum(np.square(np.linalg.norm(contact_forces, axis=1)))
        return cost
    
    def _check_health(self):
        state = self.state_vector()
        
        if not np.isfinite(state).all():
            return False, "state_not_finite", f"State values are not finite: {state}"
        
        if self._episode_count < 100:
            return True, "not_terminated", "Early learning phase"
        
        min_z = 0.15 if self.curriculum_factor < 0.5 else 0.20
        if state[2] < min_z:
            return False, "body_too_low", f"Body height: {state[2]:.3f}m < {min_z:.2f}m"
        
        w, x, y, z = state[3:7]
        roll, pitch, _ = self.euler_from_quaternion(w, x, y, z)
        
        max_roll = np.deg2rad(60 - 15 * self.curriculum_factor)
        if abs(roll) > max_roll:
            return False, "excessive_roll", f"Roll: {np.rad2deg(roll):.1f}° > {np.rad2deg(max_roll):.1f}°"
        
        if self.biped:
            target_pitch = np.deg2rad(-90)
            max_pitch_dev = np.deg2rad(90 - 10 * self.curriculum_factor)
            if abs(pitch - target_pitch) > max_pitch_dev:
                return False, "excessive_pitch", f"Pitch deviation: {np.rad2deg(abs(pitch - target_pitch)):.1f}°"
        
        return True, "not_terminated", "Healthy"
    
    def step(self, action):
        self._step += 1
        
        if self.curriculum_factor < 0.3:
            action = 0.7 * self._last_action + 0.3 * action
        
        front_contact_in_step = False
        if self.biped:
            if np.any(self.front_feet_contact_forces > 1.0):
                front_contact_in_step = True
                self._front_feet_touched = True
        
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        
        terminated = not self.is_healthy
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "termination_reason": "not_terminated",
            "termination_details": "No termination",
            "bipedal_success": False,
            **reward_info,
        }
        
        if terminated:
            is_ok, reason, details = self._check_health()
            if not is_ok:
                info["termination_reason"] = reason
                info["termination_details"] = details
        
        if truncated and self.biped and not self._front_feet_touched:
            info["bipedal_success"] = True
            self._success_count += 1
        
        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time
        
        self._last_action = action
        self._last_feet_contact_forces = self.feet_contact_forces.copy()
        
        return observation, reward, terminated, truncated, info
    
    @property
    def is_healthy(self):
        is_ok, _, _ = self._check_health()
        return is_ok
    
    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
            np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )
    
    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)
    
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)
    
    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)
    
    @property
    def feet_air_time_reward(self):
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        
        if self.biped:
            rear_feet_contact = curr_contact[2:]
            is_alternating = (rear_feet_contact[0] != rear_feet_contact[1])
            return float(is_alternating)
        
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact
        
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt
        
        time_since_threshold = (self._feet_air_time - 0.2).clip(min=0.0)
        air_time_reward = np.sum(np.square(time_since_threshold) * first_contact)
        
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1
        
        self._feet_air_time *= ~contact_filter
        
        return air_time_reward
    
    @property
    def healthy_reward(self):
        return self.is_healthy
    
    @property
    def non_flat_base_cost(self):
        return np.sum(np.square(self.projected_gravity[:2]))
    
    @property
    def collision_cost(self):
        return np.sum(
            1.0 * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
        )
    
    @property
    def joint_limit_cost(self):
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)
    
    @property
    def torque_cost(self):
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))
    
    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])
    
    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))
    
    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))
    
    @property
    def joint_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[6:]))
    
    @property
    def default_joint_position_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))
    
    def _calc_reward(self, action):
        rewards = 0
        costs = 0
        reward_info = {}
        
        if self.biped:
            upright_reward = self.biped_perfect_upright_reward * self.reward_weights["biped_perfect_upright"]
            forward_vel_reward = self.forward_velocity_reward * self.reward_weights["forward_velocity"]
            balance_reward = self.balance_stability_reward * self.reward_weights["balance_stability"]
            front_feet_off_reward = self.biped_front_feet_off_ground_reward * self.reward_weights["biped_front_feet_off_ground"]
            biped_upright_reward = self.biped_upright_reward * self.reward_weights["biped_upright"]
            
            rewards += upright_reward + forward_vel_reward + balance_reward + front_feet_off_reward + biped_upright_reward
            
            reward_info["biped_upright_reward"] = upright_reward
            reward_info["forward_velocity_reward"] = forward_vel_reward
            reward_info["balance_stability_reward"] = balance_reward
            reward_info["front_feet_off_ground_reward"] = front_feet_off_reward
        
        linear_vel_tracking_reward = self.linear_velocity_tracking_reward * self.reward_weights["linear_vel_tracking"]
        angular_vel_tracking_reward = self.angular_velocity_tracking_reward * self.reward_weights["angular_vel_tracking"]
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_reward = self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        
        rewards += linear_vel_tracking_reward + angular_vel_tracking_reward + healthy_reward + feet_air_reward
        
        reward_info["linear_vel_tracking_reward"] = linear_vel_tracking_reward
        reward_info["reward_ctrl"] = 0
        reward_info["reward_survive"] = healthy_reward
        
        adaptation_factor = 1.0 - 0.3 * self.curriculum_factor
        
        ctrl_cost = self.torque_cost * self.cost_weights["torque"] * adaptation_factor
        action_rate_cost_val = self.action_rate_cost(action) * self.cost_weights["action_rate"] * adaptation_factor
        vertical_vel_cost = self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        xy_angular_vel_cost = self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost_val = self.joint_velocity_cost * self.cost_weights["joint_velocity"] * adaptation_factor
        joint_acceleration_cost = self.acceleration_cost * self.cost_weights["joint_acceleration"]
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost * self.cost_weights["default_joint_position"]
        )
        
        costs = (
            ctrl_cost + action_rate_cost_val + vertical_vel_cost + xy_angular_vel_cost +
            joint_limit_cost + joint_velocity_cost_val + joint_acceleration_cost +
            collision_cost
        )
        
        if self.biped:
            biped_cost_scale = 0.5 + 0.5 * self.curriculum_factor
            
            front_contact_cost = self.biped_front_contact_cost * self.cost_weights["biped_front_contact"] * biped_cost_scale
            rear_feet_airborne_cost = 0.0
            if np.all(self.feet_contact_forces[2:] < 1.0):
                rear_feet_airborne_cost = self.cost_weights["biped_rear_feet_airborne"]
            front_foot_height_cost = self.biped_front_foot_height_cost * self.cost_weights["biped_front_foot_height"]
            crossed_legs_cost = self.biped_crossed_legs_cost * self.cost_weights["biped_crossed_legs"] * biped_cost_scale
            low_rear_hips_cost = self.biped_low_rear_hips_cost * self.cost_weights["biped_low_rear_hips"] * biped_cost_scale
            front_feet_below_hips_cost = self.biped_front_feet_below_hips_cost * self.cost_weights["biped_front_feet_below_hips"]
            abduction_joints_cost = self.biped_abduction_joints_cost * self.cost_weights["biped_abduction_joints"]
            unwanted_contact_cost = self.biped_unwanted_contact_cost * self.cost_weights["biped_unwanted_contact"]
            self_collision_cost_val = self.self_collision_cost * self.cost_weights["self_collision"]
            body_height_cost = self.biped_body_height_cost * self.cost_weights["biped_body_height"]
            roll_stability_cost = self.biped_roll_stability_cost * self.cost_weights["biped_roll_stability"]
            pitch_stability_cost = self.biped_pitch_stability_cost * self.cost_weights["biped_pitch_stability"]
            
            costs += (
                front_contact_cost + rear_feet_airborne_cost + front_foot_height_cost +
                crossed_legs_cost + low_rear_hips_cost + front_feet_below_hips_cost +
                abduction_joints_cost + unwanted_contact_cost + self_collision_cost_val +
                body_height_cost + roll_stability_cost + pitch_stability_cost
            )
            
            reward_info["biped_front_contact_cost"] = -front_contact_cost
            reward_info["biped_rear_feet_airborne_cost"] = -rear_feet_airborne_cost
            reward_info["biped_front_foot_height_cost"] = -front_foot_height_cost
            reward_info["biped_crossed_legs_cost"] = -crossed_legs_cost
            reward_info["biped_low_rear_hips_cost"] = -low_rear_hips_cost
            reward_info["biped_front_feet_below_hips_cost"] = -front_feet_below_hips_cost
            reward_info["biped_abduction_joints_cost"] = -abduction_joints_cost
            reward_info["biped_unwanted_contact_cost"] = -unwanted_contact_cost
            reward_info["self_collision_cost"] = -self_collision_cost_val
            reward_info["biped_body_height_cost"] = -body_height_cost
            reward_info["biped_roll_stability_cost"] = -roll_stability_cost
            reward_info["biped_pitch_stability_cost"] = -pitch_stability_cost
        else:
            costs += orientation_cost + default_joint_position_cost
            reward_info["orientation_cost"] = -orientation_cost
            reward_info["default_joint_position_cost"] = -default_joint_position_cost
        
        reward_info["reward_ctrl"] = -ctrl_cost
        
        reward = rewards - costs
        reward = max(0.0, reward) * (1.0 + 0.2 * self.curriculum_factor)
        
        return reward, reward_info
    
    def _get_obs(self):
        """48차원 관찰 공간 유지 - 원본과 동일"""
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]
        
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]
        
        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity
        
        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],  # 3
                base_angular_velocity * self._obs_scale["angular_velocity"],  # 3
                projected_gravity,  # 3
                desired_vel * self._obs_scale["linear_velocity"],  # 3
                dofs_position * self._obs_scale["dofs_position"],  # 12
                dofs_velocity * self._obs_scale["dofs_velocity"],  # 12
                last_action,  # 12
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)
        
        return curr_obs  # 총 48차원
    
    def reset_model(self):
        qpos = self.model.key_qpos[0].copy()
        
        if self.biped:
            if random.random() < 0.5:
                qpos[7:] = self.BIPEDAL_READY_JOINTS
                qpos[2] = 0.60
                
                pitch_angle = np.deg2rad(-90)
                quat = np.array([
                    np.cos(pitch_angle / 2),
                    0,
                    np.sin(pitch_angle / 2),
                    0
                ])
                qpos[3:7] = quat
        
        if self._rand_power > 0.0:
            noise_scale = 0.05 + 0.05 * self.curriculum_factor
            joint_noise = np.random.normal(
                loc=0.0,
                scale=noise_scale * self._rand_power,
                size=qpos[7:].shape
            )
            qpos[7:] += joint_noise
            joint_limits = self.model.jnt_range[1:, :]
            qpos[7:] = np.clip(qpos[7:], joint_limits[:, 0], joint_limits[:, 1])
        
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qpos[7:].copy()
        
        if self.biped:
            vel_scale = min(1.0, 0.3 + 0.7 * self.curriculum_factor)
            self._desired_velocity = np.array([0.2 * vel_scale, 0, 0])
        else:
            self._desired_velocity = self._sample_desired_vel()
        
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self._front_feet_touched = False
        self._last_feet_contact_forces = np.zeros(4)
        self._balance_history = []
        
        self._episode_count += 1
        
        observation = self._get_obs()
        return observation
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "episode_count": self._episode_count,
            "success_count": self._success_count,
            "success_rate": self._success_count / max(1, self._episode_count),
        }
    
    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel
    
    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z