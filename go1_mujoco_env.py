# 4족보행 로봇을 이용한 실험적 2족보행 훈련 코드드
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np
from pathlib import Path
import time
import random

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
    
    def __init__(self, ctrl_type="torque", biped=False, rand_power=0.0, action_noise=0.0, **kwargs):
        model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml")
        self.biped = biped
        self._rand_power = rand_power
        
        # 노이즈 추가
        self._action_noise_scale = action_noise
        self._time_since_last_noise = 0.0
        # 노이즈 추가

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
            "flipped_over": 150.0,  # 몸통 뒤집힘 비용
        }
        
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
        self.cost_weights["shoulder_below_pelvis"] = 100.0  # 어깨가 골반보다 낮을 때 비용
        self.cost_weights["hip_ground_contact"] = 200.0  # hip이 땅에 닿을 때 비용
            
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
        
        self._healthy_z_range = (0.30, 1.8)
        self._healthy_pitch_range = (-np.pi, 0.0)
        self._healthy_roll_range = (-np.deg2rad(85), np.deg2rad(85))
        
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]
        self._cfrc_ext_front_feet_indices = [4, 7]
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
        
        dof_position_limit_multiplier = 0.95
        ctrl_range_offset = (
            0.5 * (1 - dof_position_limit_multiplier) * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
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
        

        self._initialize_biped_body_ids()

        self._time_flipped_over = 0.0 # 몸통 뒤집힘 타이머
        self._time_shoulder_below_pelvis = 0.0  # 어깨가 골반보다 낮을 때 타이머
        self._time_hip_on_ground = 0.0  # hip이 땅에 닿을 때 타이머
        self._time_trunk_low = 0.0  # 몸통 높이가 낮을 때 타이머
    
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
    def _trunk_up_alignment(self):
        """Helper to get alignment of trunk's z-axis with world's up-axis."""
        world_up_vector = np.array([0, 0, 1])
        # 몸통의 z-축이 월드의 up-축과 얼마나 정렬되어 있는지 계산합니다.
        trunk_up_vector = self.data.xmat[self._main_body_id].reshape(3, 3)[:, 2]
        return np.dot(trunk_up_vector, world_up_vector)

    @property
    def is_flipped_over(self):
        """Check if the robot is flipped over."""
        # 몸통의 z-축이 월드의 up-축과 얼마나 정렬되어 있는지 계산합니다.
        return self._trunk_up_alignment < 0.0

    @property
    def flipped_over_cost(self):
        """Calculate cost for being flipped over."""
        alignment = self._trunk_up_alignment
        if alignment < 0:
            # 몸통이 뒤집힌 경우, alignment는 0에서 -1 사이의 값을 가집니다.
            # 몸통이 뒤집힌 경우, alignment는 0에서 -1 사이의 값을 가집니다.
            return np.square(alignment)
        return 0.0


    @property
    def forward_velocity_reward(self):
        forward_vel = self.data.qvel[0]
        target_vel = 0.2
        vel_error = abs(forward_vel - target_vel)
        reward = np.exp(-5.0 * vel_error)
        if forward_vel < 0:
            reward *= 0.1
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
        if z_pos < self._healthy_z_range[0]:
            penalty = np.exp((self._healthy_z_range[0] - z_pos) * 30) - 1
            return penalty
        return 0.0
        
    @property
    def is_trunk_low(self):
        """몸통 높이가 낮은지 확인합니다."""
        z_pos = self.data.qpos[2]
        return z_pos < self._healthy_z_range[0]  # 0.25m 미만일 때 낮다고 판단
    
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
    def shoulder_below_pelvis_cost(self):
        """어깨가 골반보다 낮은 경우 비용을 계산합니다."""
        if not self.biped:
            return 0.0
        
        # 어깨 위치 (FR_hip, FL_hip의 평균)
        shoulder_pos = self.data.xpos[self._front_hip_body_ids]
        shoulder_z = np.mean(shoulder_pos[:, 2])
        
        # 골반 위치 (RR_hip, RL_hip의 평균)
        pelvis_pos = self.data.xpos[self._rear_hip_body_ids]
        pelvis_z = np.mean(pelvis_pos[:, 2])
        
        # 어깨가 골반보다 낮을 때의 높이 차이
        height_difference = pelvis_z - shoulder_z
        
        if height_difference > 0:  # 어깨가 골반보다 낮음
            # 높이 차이가 클수록 비용이 급격히 증가
            return np.square(height_difference) * 10.0
        return 0.0
    
    @property
    def is_shoulder_below_pelvis(self):
        """어깨가 골반보다 낮은지 확인합니다."""
        if not self.biped:
            return False
        
        shoulder_pos = self.data.xpos[self._front_hip_body_ids]
        shoulder_z = np.mean(shoulder_pos[:, 2])
        
        pelvis_pos = self.data.xpos[self._rear_hip_body_ids]
        pelvis_z = np.mean(pelvis_pos[:, 2])
        
        return shoulder_z < pelvis_z
    
    @property
    def is_hip_on_ground(self):
        """hip이 땅에 닿을 때 확인합니다."""
        # hip body의 위치를 확인합니다.
        all_hip_body_ids = self._front_hip_body_ids + self._rear_hip_body_ids
        hip_contact_forces = self.data.cfrc_ext[all_hip_body_ids]
        
        # hip이 땅에 닿을 때의 임계값
        contact_threshold = 0.1
        return np.any(np.linalg.norm(hip_contact_forces, axis=1) > contact_threshold)
    
    @property
    def hip_ground_contact_cost(self):
        """hip이 땅에 닿을 때의 비용을 계산합니다."""
        if not self.is_hip_on_ground:
            return 0.0
        
        # hip이 땅에 닿을 때의 임계값
        all_hip_body_ids = self._front_hip_body_ids + self._rear_hip_body_ids
        hip_contact_forces = self.data.cfrc_ext[all_hip_body_ids]
        
        # hip이 땅에 닿을 때의 비용을 계산합니다.
        contact_magnitudes = np.linalg.norm(hip_contact_forces, axis=1)
        active_contacts = contact_magnitudes > 0.1
        
        if not np.any(active_contacts):
            return 0.0
        
        # hip이 땅에 닿을 때의 비용을 계산합니다.
        active_contact_forces = contact_magnitudes[active_contacts]
        cost = np.sum(np.square(active_contact_forces))
        
        return cost
    
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
        
        # 몸통이 뒤집힌 경우 타이머가 1초 이상 지속되면 에피소드 종료
        if self._time_flipped_over > 1.0:
            return False, "flipped_over_timeout", f"Flipped over for {self._time_flipped_over:.2f}s > 1.0s"
        
        # 어깨가 골반보다 낮은 경우 타이머가 1초 이상 지속되면 에피소드 종료
        if self._time_shoulder_below_pelvis > 1.0:
            return False, "shoulder_below_pelvis_timeout", f"Shoulder below pelvis for {self._time_shoulder_below_pelvis:.2f}s > 1.0s"
        
        # hip이 땅에 닿을 때 타이머가 1초 이상 지속되면 에피소드 종료
        if self._time_hip_on_ground > 1.0:
            return False, "hip_ground_contact_timeout", f"Hip on ground for {self._time_hip_on_ground:.2f}s > 1.0s"
        
        # 몸통 높이가 낮을 때 타이머가 1초 이상 지속되면 에피소드 종료
        if self._time_trunk_low > 0.36:
            return False, "trunk_low_timeout", f"Trunk height low for {self._time_trunk_low:.2f}s > 0.4s"
        
        


        return True, "not_terminated", "Healthy"
    
    def step(self, action):
        self._step += 1
        
        if self.curriculum_factor < 0.3:
            action = 0.7 * self._last_action + 0.3 * action
        
        # 노이즈 추가
        # 0.5초마다 노이즈 추가
        self._time_since_last_noise += self.dt
        if self._time_since_last_noise > 0.5:#True:#self._action_noise_scale > 0.0 and self._time_since_last_noise > 0.5:
            # 노이즈 추가
            #print("noise added")
            current_noise_level = self._action_noise_scale * self.curriculum_factor
            # action에 노이즈 추가
            noise = np.random.normal(0, current_noise_level, size=action.shape)
            action = action + noise 
            # action 값을 [-1, 1] 범위로 제한
            #action = np.clip(action, -1.0, 1.0)
            # 노이즈 추가 타이머 초기화
            self._time_since_last_noise = 0.0
        # 노이즈 추가

        front_contact_in_step = False
        if np.any(self.front_feet_contact_forces > 1.0):
            front_contact_in_step = True
            self._front_feet_touched = True
        
        self.do_simulation(action, self.frame_skip)
        
        # 몸통이 뒤집힌 경우 타이머 증가
        if self.is_flipped_over:
            self._time_flipped_over += self.dt
        else:
            self._time_flipped_over = 0.0
        
        # 어깨가 골반보다 낮은 경우 타이머 증가
        if self.is_shoulder_below_pelvis:
            self._time_shoulder_below_pelvis += self.dt
        else:
            self._time_shoulder_below_pelvis = 0.0
        
        # hip이 땅에 닿을 때 타이머 증가
        if self.is_hip_on_ground:
            self._time_hip_on_ground += self.dt
        else:
            self._time_hip_on_ground = 0.0
            
        # 몸통 높이가 낮을 때 타이머 증가
        if self.is_trunk_low:
            self._time_trunk_low += self.dt
        else:
            self._time_trunk_low = 0.0
            
        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        
        # 보상 계산
        self._current_episode_reward = getattr(self, '_current_episode_reward', 0.0) + reward
        self._current_episode_length = getattr(self, '_current_episode_length', 0) + 1
        
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
        
        # 몸통이 뒤집힌 경우 비용 추가
        flipped_cost = self.flipped_over_cost * self.cost_weights["flipped_over"]
        reward_info["flipped_over_cost"] = -flipped_cost

        costs = (
            ctrl_cost + action_rate_cost_val + vertical_vel_cost + xy_angular_vel_cost +
            joint_limit_cost + joint_velocity_cost_val + joint_acceleration_cost +
            collision_cost + flipped_cost # 몸통이 뒤집힌 경우 비용 추가
        )
        

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
        
        # 어깨가 골반보다 낮은 경우 비용 추가
        shoulder_below_pelvis_cost = self.shoulder_below_pelvis_cost * self.cost_weights["shoulder_below_pelvis"]
        
        # hip이 땅에 닿을 때 비용 추가
        hip_ground_contact_cost = self.hip_ground_contact_cost * self.cost_weights["hip_ground_contact"]
        
        costs += (
            front_contact_cost + rear_feet_airborne_cost + front_foot_height_cost +
            crossed_legs_cost + low_rear_hips_cost + front_feet_below_hips_cost +
            abduction_joints_cost + unwanted_contact_cost + self_collision_cost_val +
            body_height_cost + roll_stability_cost + pitch_stability_cost +
            shoulder_below_pelvis_cost + hip_ground_contact_cost
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
        reward_info["shoulder_below_pelvis_cost"] = -shoulder_below_pelvis_cost
        reward_info["hip_ground_contact_cost"] = -hip_ground_contact_cost

        reward_info["reward_ctrl"] = -ctrl_cost
        
        reward = rewards - costs
        reward = max(0.0, reward) * (1.0 + 0.2 * self.curriculum_factor)
        
        return reward, reward_info
    
    def _get_obs(self):
        """48개의 관측 변수를 반환합니다."""
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
        
        return curr_obs  # 48개의 관측 변수
    
    def reset_model(self):
        qpos = self.model.key_qpos[0].copy()
        
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
        
   
        vel_scale = min(1.0, 0.3 + 0.7 * self.curriculum_factor)
        self._desired_velocity = np.array([0.2 * vel_scale, 0, 0])
        
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self._front_feet_touched = False
        self._last_feet_contact_forces = np.zeros(4)
        self._balance_history = []
        self._time_flipped_over = 0.0 # 몸통이 뒤집힌 경우 타이머
        self._time_shoulder_below_pelvis = 0.0  # 어깨가 골반보다 낮은 경우 타이머
        self._time_hip_on_ground = 0.0  # hip이 땅에 닿을 때 타이머
        self._time_trunk_low = 0.0  # 몸통 높이가 낮은 경우 타이머
        
        # 보상 계산
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

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
    
    # 보상 계산
    def get_detailed_episode_info(self):
        """보상 계산"""
        try:
            # 에피소드 정보
            episode_info = {
                'episode_count': self._episode_count,
                'success_count': self._success_count,
                'success_rate': self._success_count / max(1, self._episode_count),
                'current_step': self._step,
                'max_episode_time': self._max_episode_time_sec,
                'time_remaining': max(0, self._max_episode_time_sec - self._step * self.dt),
            }
            
            # 로봇 상태
            robot_state = {
                'position': {
                    'x': float(self.data.qpos[0]),
                    'y': float(self.data.qpos[1]),
                    'z': float(self.data.qpos[2]),
                    'distance_from_origin': float(np.linalg.norm(self.data.qpos[0:2], ord=2))
                },
                'orientation': {
                    'quaternion': self.data.qpos[3:7].tolist(),
                    'euler': self.euler_from_quaternion(*self.data.qpos[3:7]).tolist()
                },
                'velocity': {
                    'linear': self.data.qvel[:3].tolist(),
                    'angular': self.data.qvel[3:6].tolist(),
                    'joint': self.data.qvel[6:].tolist()
                },
                'joint_positions': self.data.qpos[7:].tolist(),
                'joint_velocities': self.data.qvel[6:].tolist(),
                'joint_accelerations': self.data.qacc[6:].tolist() if hasattr(self.data, 'qacc') else None,
            }
            
            # 보상 정보
            reward_info = {
                'current_episode_reward': getattr(self, '_current_episode_reward', 0.0),  # 현재 에피소드 보상
                'current_episode_length': getattr(self, '_current_episode_length', 0),    # 현재 에피소드 길이
                'reward_weights': self.reward_weights.copy(),
                'cost_weights': self.cost_weights.copy(),
            }
            
            # 환경 설정
            env_config = {
                'ctrl_type': getattr(self, 'ctrl_type', 'unknown'),
                'biped': self.biped,
                'rand_power': self._rand_power,
                'action_noise_scale': self._action_noise_scale,
                'frame_skip': self.frame_skip,
                'dt': self.dt,
                'curriculum_factor': self.curriculum_factor,
                'desired_velocity': self._desired_velocity.tolist(),
                'healthy_z_range': self._healthy_z_range,
                'healthy_pitch_range': self._healthy_pitch_range,
                'healthy_roll_range': self._healthy_roll_range,
            }
            
            # 접촉 정보
            contact_info = {
                'feet_contact_forces': self.feet_contact_forces.tolist(),
                'front_feet_contact_forces': self.front_feet_contact_forces.tolist(),
                'feet_air_time': self._feet_air_time.tolist(),
                'last_contacts': self._last_contacts.tolist(),
            }
            
            # 안정성 정보
            stability_metrics = {
                'is_healthy': self.is_healthy,
                'is_flipped_over': self.is_flipped_over,
                'time_flipped_over': self._time_flipped_over,
                'is_shoulder_below_pelvis': self.is_shoulder_below_pelvis,
                'time_shoulder_below_pelvis': self._time_shoulder_below_pelvis,
                'is_hip_on_ground': self.is_hip_on_ground,
                'time_hip_on_ground': self._time_hip_on_ground,
                'is_trunk_low': self.is_trunk_low,
                'time_trunk_low': self._time_trunk_low,
                'shoulder_below_pelvis_cost': float(self.shoulder_below_pelvis_cost),
                'hip_ground_contact_cost': float(self.hip_ground_contact_cost),
                'trunk_up_alignment': float(self._trunk_up_alignment),
                'projected_gravity': self.projected_gravity.tolist(),
                'balance_history': self._balance_history.copy() if hasattr(self, '_balance_history') else [],
            }
            
            # 액션 정보
            action_info = {
                'last_action': self._last_action.tolist(),
                'last_feet_contact_forces': self._last_feet_contact_forces.tolist(),
            }
            
            return {
                'episode_info': episode_info,
                'robot_state': robot_state,
                'reward_info': reward_info,
                'env_config': env_config,
                'contact_info': contact_info,
                'stability_metrics': stability_metrics,
                'action_info': action_info,
                'timestamp': time.time(),
                'simulation_time': self.data.time,
            }
            
        except Exception as e:
            # 오류 정보
            return {
                'episode_info': {
                    'episode_count': self._episode_count,
                    'success_count': self._success_count,
                    'error': str(e)
                },
                'timestamp': time.time(),
                'error': f"오류: {str(e)}"
            }
    
    def get_environment_summary(self):
        """환경 정보"""
        try:
            return {
                'environment_class': self.__class__.__name__,
                'model_path': str(getattr(self, 'model_path', 'unknown')),
                'observation_space': str(self.observation_space),
                'action_space': str(self.action_space),
                'metadata': self.metadata,
                'frame_skip': self.frame_skip,
                'dt': self.dt,
                'max_episode_time_sec': self._max_episode_time_sec,
                'reset_noise_scale': self._reset_noise_scale,
                'clip_obs_threshold': self._clip_obs_threshold,
                'max_balance_history': self._max_balance_history,
                'curriculum_base': self._curriculum_base,
                'tracking_velocity_sigma': self._tracking_velocity_sigma,
                'desired_velocity_min': self._desired_velocity_min.tolist(),
                'desired_velocity_max': self._desired_velocity_max.tolist(),
                'obs_scale': self._obs_scale,
                'gravity_vector': self._gravity_vector.tolist(),
                'default_joint_position': self._default_joint_position.tolist(),
                'bipedal_ready_joints': self.BIPEDAL_READY_JOINTS.tolist(),
                'feet_site_names': list(self._feet_site_name_to_id.keys()),
                'main_body_id': self._main_body_id,
                'cfrc_ext_feet_indices': self._cfrc_ext_feet_indices,
                'cfrc_ext_front_feet_indices': self._cfrc_ext_front_feet_indices,
                'cfrc_ext_contact_indices': self._cfrc_ext_contact_indices,
                'timestamp': time.time(),
            }
        except Exception as e:
            return {
                'environment_class': self.__class__.__name__,
                'error': f"오류: {str(e)}",
                'timestamp': time.time(),
            }
    
    def get_performance_metrics(self):
        """성능 지표"""
        try:
            # 보상 구성 요소
            reward_components = {}
            if hasattr(self, 'reward_weights'):
                for key in self.reward_weights:
                    if hasattr(self, key):
                        try:
                            reward_components[key] = float(getattr(self, key))
                        except:
                            reward_components[key] = 0.0
            
            # 비용 구성 요소
            cost_components = {}
            if hasattr(self, 'cost_weights'):
                for key in self.cost_weights:
                    if hasattr(self, key):
                        try:
                            cost_components[key] = float(getattr(self, key))
                        except:
                            cost_components[key] = 0.0
            
            # 안정성 점수
            stability_score = 0.0
            if hasattr(self, '_balance_history') and self._balance_history:
                stability_score = float(np.mean(self._balance_history))
            
            # 진행 상태
            progress = min(1.0, self._step / (self._max_episode_time_sec / self.dt))
            
            return {
                'episode_progress': progress,
                'current_step': self._step,
                'max_steps': int(self._max_episode_time_sec / self.dt),
                'reward_components': reward_components,
                'cost_components': cost_components,
                'stability_score': stability_score,
                'is_healthy': self.is_healthy,
                'is_flipped_over': self.is_flipped_over,
                'time_flipped_over': self._time_flipped_over,
                'is_shoulder_below_pelvis': self.is_shoulder_below_pelvis,
                'time_shoulder_below_pelvis': self._time_shoulder_below_pelvis,
                'is_hip_on_ground': self.is_hip_on_ground,
                'time_hip_on_ground': self._time_hip_on_ground,
                'is_trunk_low': self.is_trunk_low,
                'time_trunk_low': self._time_trunk_low,
                'curriculum_factor': self.curriculum_factor,
                'timestamp': time.time(),
            }
        except Exception as e:
            return {
                'error': f"오류: {str(e)}",
                'timestamp': time.time(),
            }   