#4족보행 로봇을 이용한 실험적 2족보행 훈련 코드드
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
        
        self._action_noise_scale = action_noise
        self._time_since_last_noise = 0.0

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
        
        # ==========================================
        # 정규화된 보상 가중치 (최대: 10.0)
        # ==========================================
        self.reward_weights = {
            # 핵심 목표 (7-10 범위)
            "biped_perfect_upright": 10.0,       # 최우선: 완벽한 직립 자세
            "balance_stability": 8.0,            # 균형 안정성
            "biped_front_feet_off_ground": 8.0,  # 앞발 들기 (핵심!)
            "forward_velocity": 7.0,             # 전진 속도
            
            # 보조 목표 (3-5 범위)
            "biped_upright": 5.0,                # 몸통 직립 정렬
            "feet_airtime": 4.0,                 # 발 체공 시간
            
            # 기본 보상 (0.5-2 범위)
            "linear_vel_tracking": 1.0,          # 선속도 추적
            "angular_vel_tracking": 0.8,         # 각속도 추적
            "healthy": 0.5,                      # 생존 보상
        }
        
        # ==========================================
        # 정규화된 비용 가중치 (최대: 10.0, 최소: 0.01)
        # ==========================================
        self.cost_weights = {
            # 레벨 1: 치명적 실패 (8-10 범위)
            "flipped_over": 10.0,                # 뒤집힘 - 최대 페널티
            "hip_ground_contact": 9.0,           # hip이 땅에 닿음
            "shoulder_below_pelvis": 8.0,        # 어깨가 골반보다 낮음
            
            # 레벨 2: 주요 실패 (4-7 범위)
            "biped_unwanted_contact": 6.0,       # 원치 않는 접촉
            "biped_front_feet_below_hips": 5.0,  # 앞발이 엉덩이보다 낮음
            "biped_front_contact": 4.0,          # 앞발 접촉 (기존 80에서 대폭 감소)
            "joint_limit": 4.0,                  # 관절 한계
            
            # 레벨 3: 자세 제약 (1-3 범위)
            "self_collision": 3.0,               # 자가 충돌
            "biped_pitch_stability": 2.0,        # 피치 안정성
            "biped_roll_stability": 2.0,         # 롤 안정성
            "biped_low_rear_hips": 1.5,         # 낮은 뒷다리 엉덩이
            "biped_body_height": 1.5,           # 몸체 높이
            "collision": 1.0,                    # 일반 충돌
            
            # 레벨 4: 부드러운 동작 (0.1-0.9 범위)
            "biped_front_foot_height": 0.8,      # 앞발 높이
            "biped_crossed_legs": 0.6,          # 다리 교차
            "biped_rear_feet_airborne": 0.5,    # 뒷발 체공
            "vertical_vel": 0.4,                # 수직 속도
            "orientation": 0.3,                  # 방향
            
            # 레벨 5: 미세 조정 (0.01-0.09 범위)
            "biped_abduction_joints": 0.08,     # 외전 관절
            "xy_angular_vel": 0.05,             # XY 각속도
            "default_joint_position": 0.05,     # 기본 관절 위치
            "joint_velocity": 0.03,             # 관절 속도
            "action_rate": 0.02,                # 액션 변화율
            "torque": 0.01,                     # 토크 (최소값)
            "joint_acceleration": 0.01,         # 관절 가속도 (최소값)
        }
            
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

        self._time_flipped_over = 0.0
        self._time_shoulder_below_pelvis = 0.0
        self._time_hip_on_ground = 0.0
        
        # ==========================================
        # 보상 정규화를 위한 추가 변수
        # ==========================================
        self._reward_normalizer_window = 1000
        self._reward_history = []
        self._cost_history = []
        
        # 스케일 검증 출력
        self._validate_weight_scales()

    def _validate_weight_scales(self):
        """가중치 스케일 범위를 검증하고 출력합니다."""
        max_reward = max(self.reward_weights.values())
        min_reward = min(self.reward_weights.values())
        max_cost = max(self.cost_weights.values())
        min_cost = min(self.cost_weights.values())
        
        print("=" * 50)
        print("📊 보상/비용 가중치 스케일 분석")
        print("=" * 50)
        print(f"✅ 보상 범위: {min_reward:.2f} ~ {max_reward:.2f} (비율 {max_reward/min_reward:.1f}:1)")
        print(f"✅ 비용 범위: {min_cost:.2f} ~ {max_cost:.2f} (비율 {max_cost/min_cost:.1f}:1)")
        print(f"✅ 전체 동적 범위: {max_reward/min_cost:.0f}:1")
        print("=" * 50)
        
        # 경고 체크
        if max_reward > 10.0:
            print("⚠️ 경고: 최대 보상이 10.0을 초과합니다!")
        if min_cost < 0.01:
            print("⚠️ 경고: 최소 비용이 0.01 미만입니다!")
        if max_reward/min_cost > 1000:
            print("⚠️ 경고: 동적 범위가 1000:1을 초과합니다!")

    def _apply_curriculum_scaling(self, weight_dict):
        """커리큘럼 학습 진행도에 따라 가중치를 동적으로 조정합니다."""
        scaled_weights = weight_dict.copy()
        curriculum = self.curriculum_factor
        
        if curriculum < 0.3:  # 초기 단계: 안정성 중심
            stability_keys = ["balance_stability", "biped_perfect_upright", "flipped_over", "hip_ground_contact"]
            for key in stability_keys:
                if key in scaled_weights:
                    scaled_weights[key] *= 1.3
                    
        elif curriculum < 0.7:  # 중간 단계: 동작 학습
            motion_keys = ["biped_front_feet_off_ground", "forward_velocity", "biped_front_contact"]
            for key in motion_keys:
                if key in scaled_weights:
                    if "contact" in key:  # 접촉 페널티는 감소
                        scaled_weights[key] *= 0.7
                    else:  # 동작 보상은 증가
                        scaled_weights[key] *= 1.2
                        
        else:  # 후기 단계: 성능 최적화
            performance_keys = ["forward_velocity", "feet_airtime", "torque", "action_rate"]
            for key in performance_keys:
                if key in scaled_weights:
                    if key in ["torque", "action_rate"]:  # 효율성 페널티 감소
                        scaled_weights[key] *= 0.5
                    else:  # 성능 보상 증가
                        scaled_weights[key] *= 1.3
        
        return scaled_weights

    def _normalize_reward_value(self, value, max_val=10.0):
        """보상 값을 정규화합니다 (tanh 기반 부드러운 포화)."""
        if abs(value) < 0.001:
            return 0.0
        return np.tanh(value / max_val) * max_val
    
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
        # 몸통 회전 행렬의 3번째 열이 몸통의 Z축(Up 벡터)을 월드 좌표계 기준으로 나타냅니다.
        trunk_up_vector = self.data.xmat[self._main_body_id].reshape(3, 3)[:, 2]
        return np.dot(trunk_up_vector, world_up_vector)

    @property
    def is_flipped_over(self):
        """Check if the robot is flipped over."""
        # 몸통의 Up 벡터와 세상의 Up 벡터의 내적 값이 음수이면 뒤집힌 상태입니다.
        return self._trunk_up_alignment < 0.0

    @property
    def flipped_over_cost(self):
        """Calculate cost for being flipped over."""
        alignment = self._trunk_up_alignment
        if alignment < 0:
            # 뒤집혔을 때, alignment 값은 0 ~ -1 사이입니다.
            # 제곱을 하여 -1에 가까울수록 (완전히 뒤집힐수록) 비용이 1에 가깝게 증가합니다.
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
    def shoulder_below_pelvis_cost(self):
        """어깨가 골반보다 낮을 때의 비용을 계산합니다."""
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
        """hip이 땅에 닿는지 확인합니다."""
        # 모든 hip body들의 접촉 힘을 확인
        all_hip_body_ids = self._front_hip_body_ids + self._rear_hip_body_ids
        hip_contact_forces = self.data.cfrc_ext[all_hip_body_ids]
        
        # 접촉 힘의 크기가 임계값(0.1)을 초과하면 땅에 닿은 것으로 판단
        contact_threshold = 0.1
        return np.any(np.linalg.norm(hip_contact_forces, axis=1) > contact_threshold)
    
    @property
    def hip_ground_contact_cost(self):
        """hip이 땅에 닿을 때의 비용을 계산합니다."""
        if not self.is_hip_on_ground:
            return 0.0
        
        # hip이 땅에 닿은 경우, 접촉 힘의 크기에 비례하여 비용 계산
        all_hip_body_ids = self._front_hip_body_ids + self._rear_hip_body_ids
        hip_contact_forces = self.data.cfrc_ext[all_hip_body_ids]
        
        # 접촉 힘의 크기를 계산하고 제곱하여 비용 증가
        contact_magnitudes = np.linalg.norm(hip_contact_forces, axis=1)
        active_contacts = contact_magnitudes > 0.1
        
        if not np.any(active_contacts):
            return 0.0
        
        # 활성 접촉에 대해서만 비용 계산
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
        
        # 뒤집힌 상태가 1초 이상 지속되면 에피소드 종료 
        if self._time_flipped_over > 1.0:
            return False, "flipped_over_timeout", f"Flipped over for {self._time_flipped_over:.2f}s > 1.0s"
        
        # 6. 어깨가 골반보다 낮은 상태가 1초 이상 지속되면 에피소드 종료
        if self._time_shoulder_below_pelvis > 1.0:
            return False, "shoulder_below_pelvis_timeout", f"Shoulder below pelvis for {self._time_shoulder_below_pelvis:.2f}s > 1.0s"
        
        # 7. hip이 땅에 닿은 상태가 1초 이상 지속되면 에피소드 종료
        if self._time_hip_on_ground > 1.0:
            return False, "hip_ground_contact_timeout", f"Hip on ground for {self._time_hip_on_ground:.2f}s > 1.0s"
        
        return True, "not_terminated", "Healthy"
    
    def step(self, action):
        self._step += 1
        
        #if self.curriculum_factor < 0.3:
         #   action = 0.7 * self._last_action + 0.3 * action
        
        # ✨ --- 수정된 부분 시작 --- ✨
        # 0.5초마다 커리큘럼에 따라 강도가 조절된 노이즈를 action에 추가합니다.
        self._time_since_last_noise += self.dt
        if self._time_since_last_noise > 0.5:#True:#self._action_noise_scale > 0.0 and self._time_since_last_noise > 0.5:
            # 커리큘럼에 따라 현재 노이즈 레벨을 계산합니다.
            #print("noise added")
            current_noise_level = self._action_noise_scale * self.curriculum_factor
            # 노이즈를 생성하고 action에 더합니다.
            noise = np.random.normal(0, current_noise_level, size=action.shape)
            action = action + noise 
            # action 값이 유효 범위 [-1, 1]을 벗어나지 않도록 클리핑합니다.
            #action = np.clip(action, -1.0, 1.0)
            # 타이머를 리셋합니다.
            self._time_since_last_noise = 0.0
        # ✨ --- 수정된 부분 끝 --- ✨

        front_contact_in_step = False
        if np.any(self.front_feet_contact_forces > 1.0):
            front_contact_in_step = True
            self._front_feet_touched = True
        
        self.do_simulation(action, self.frame_skip)
        
        #  매 스텝마다 뒤집힘 상태를 확인하고 타이머 업데이트 
        if self.is_flipped_over:
            self._time_flipped_over += self.dt
        else:
            self._time_flipped_over = 0.0
        
        # 5. 매 스텝마다 어깨-골반 높이 상태를 확인하고 타이머 업데이트
        if self.is_shoulder_below_pelvis:
            self._time_shoulder_below_pelvis += self.dt
        else:
            self._time_shoulder_below_pelvis = 0.0
        
        # 6. 매 스텝마다 hip 접촉 상태를 확인하고 타이머 업데이트
        if self.is_hip_on_ground:
            self._time_hip_on_ground += self.dt
        else:
            self._time_hip_on_ground = 0.0
            
        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        
        # ✨ [신규 추가] 실시간 저장을 위한 보상 정보 업데이트
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
        """개선된 보상 계산 함수 (정규화된 스케일 적용)"""
        rewards = 0
        costs = 0
        reward_info = {}
        
        # 커리큘럼 기반 가중치 조정
        adjusted_reward_weights = self._apply_curriculum_scaling(self.reward_weights)
        adjusted_cost_weights = self._apply_curriculum_scaling(self.cost_weights)
        
        # ==========================================
        # 보상 계산 (정규화 적용)
        # ==========================================
        
        # 핵심 보상들
        upright_raw = self.biped_perfect_upright_reward
        upright_reward = self._normalize_reward_value(
            upright_raw * adjusted_reward_weights["biped_perfect_upright"], 10.0
        )
        
        forward_vel_raw = self.forward_velocity_reward
        forward_vel_reward = self._normalize_reward_value(
            forward_vel_raw * adjusted_reward_weights["forward_velocity"], 10.0
        )
        
        balance_raw = self.balance_stability_reward
        balance_reward = self._normalize_reward_value(
            balance_raw * adjusted_reward_weights["balance_stability"], 10.0
        )
        
        front_feet_off_raw = self.biped_front_feet_off_ground_reward
        front_feet_off_reward = self._normalize_reward_value(
            front_feet_off_raw * adjusted_reward_weights["biped_front_feet_off_ground"], 10.0
        )
        
        biped_upright_raw = self.biped_upright_reward
        biped_upright_reward = self._normalize_reward_value(
            biped_upright_raw * adjusted_reward_weights["biped_upright"], 10.0
        )
        
        # 보조 보상들
        linear_vel_tracking_reward = self._normalize_reward_value(
            self.linear_velocity_tracking_reward * adjusted_reward_weights["linear_vel_tracking"], 10.0
        )
        angular_vel_tracking_reward = self._normalize_reward_value(
            self.angular_velocity_tracking_reward * adjusted_reward_weights["angular_vel_tracking"], 10.0
        )
        healthy_reward = self._normalize_reward_value(
            self.healthy_reward * adjusted_reward_weights["healthy"], 10.0
        )
        feet_air_reward = self._normalize_reward_value(
            self.feet_air_time_reward * adjusted_reward_weights["feet_airtime"], 10.0
        )
        
        # 총 보상 합산
        rewards = (upright_reward + forward_vel_reward + balance_reward + 
                front_feet_off_reward + biped_upright_reward + linear_vel_tracking_reward + 
                angular_vel_tracking_reward + healthy_reward + feet_air_reward)
        
        # 보상 정보 저장
        reward_info["biped_upright_reward"] = upright_reward
        reward_info["forward_velocity_reward"] = forward_vel_reward
        reward_info["balance_stability_reward"] = balance_reward
        reward_info["front_feet_off_ground_reward"] = front_feet_off_reward
        reward_info["linear_vel_tracking_reward"] = linear_vel_tracking_reward
        reward_info["reward_survive"] = healthy_reward
        
        # ==========================================
        # 비용 계산 (정규화 적용)
        # ==========================================
        
        # 적응 계수 (커리큘럼에 따라 감소)
        adaptation_factor = 1.0 - 0.3 * self.curriculum_factor
        
        # 레벨 1: 치명적 실패
        flipped_cost = self._normalize_reward_value(
            self.flipped_over_cost * adjusted_cost_weights["flipped_over"], 10.0
        )
        hip_ground_cost = self._normalize_reward_value(
            self.hip_ground_contact_cost * adjusted_cost_weights["hip_ground_contact"], 10.0
        )
        shoulder_below_cost = self._normalize_reward_value(
            self.shoulder_below_pelvis_cost * adjusted_cost_weights["shoulder_below_pelvis"], 10.0
        )
        
        # 레벨 2: 주요 실패
        unwanted_contact_cost = self._normalize_reward_value(
            self.biped_unwanted_contact_cost * adjusted_cost_weights["biped_unwanted_contact"], 10.0
        )
        front_contact_cost = self._normalize_reward_value(
            self.biped_front_contact_cost * adjusted_cost_weights["biped_front_contact"], 10.0
        )
        front_feet_below_cost = self._normalize_reward_value(
            self.biped_front_feet_below_hips_cost * adjusted_cost_weights["biped_front_feet_below_hips"], 10.0
        )
        joint_limit_cost = self._normalize_reward_value(
            self.joint_limit_cost * adjusted_cost_weights["joint_limit"], 10.0
        )
        
        # 레벨 3: 자세 제약
        self_collision_cost = self._normalize_reward_value(
            self.self_collision_cost * adjusted_cost_weights["self_collision"], 10.0
        )
        pitch_stability_cost = self._normalize_reward_value(
            self.biped_pitch_stability_cost * adjusted_cost_weights["biped_pitch_stability"], 10.0
        )
        roll_stability_cost = self._normalize_reward_value(
            self.biped_roll_stability_cost * adjusted_cost_weights["biped_roll_stability"], 10.0
        )
        low_rear_hips_cost = self._normalize_reward_value(
            self.biped_low_rear_hips_cost * adjusted_cost_weights["biped_low_rear_hips"], 10.0
        )
        body_height_cost = self._normalize_reward_value(
            self.biped_body_height_cost * adjusted_cost_weights["biped_body_height"], 10.0
        )
        collision_cost = self._normalize_reward_value(
            self.collision_cost * adjusted_cost_weights["collision"], 10.0
        )
        
        # 레벨 4: 부드러운 동작
        front_foot_height_cost = self._normalize_reward_value(
            self.biped_front_foot_height_cost * adjusted_cost_weights["biped_front_foot_height"], 10.0
        )
        crossed_legs_cost = self._normalize_reward_value(
            self.biped_crossed_legs_cost * adjusted_cost_weights["biped_crossed_legs"], 10.0
        )
        rear_feet_airborne_cost = 0.0
        if np.all(self.feet_contact_forces[2:] < 1.0):
            rear_feet_airborne_cost = self._normalize_reward_value(
                adjusted_cost_weights["biped_rear_feet_airborne"], 10.0
            )
        vertical_vel_cost = self._normalize_reward_value(
            self.vertical_velocity_cost * adjusted_cost_weights["vertical_vel"], 10.0
        )
        orientation_cost = self._normalize_reward_value(
            self.non_flat_base_cost * adjusted_cost_weights["orientation"], 10.0
        )
        
        # 레벨 5: 미세 조정
        abduction_joints_cost = self._normalize_reward_value(
            self.biped_abduction_joints_cost * adjusted_cost_weights["biped_abduction_joints"], 10.0
        )
        xy_angular_vel_cost = self._normalize_reward_value(
            self.xy_angular_velocity_cost * adjusted_cost_weights["xy_angular_vel"], 10.0
        )
        default_joint_cost = self._normalize_reward_value(
            self.default_joint_position_cost * adjusted_cost_weights["default_joint_position"], 10.0
        )
        joint_velocity_cost = self._normalize_reward_value(
            self.joint_velocity_cost * adjusted_cost_weights["joint_velocity"] * adaptation_factor, 10.0
        )
        action_rate_cost = self._normalize_reward_value(
            self.action_rate_cost(action) * adjusted_cost_weights["action_rate"] * adaptation_factor, 10.0
        )
        torque_cost = self._normalize_reward_value(
            self.torque_cost * adjusted_cost_weights["torque"] * adaptation_factor, 10.0
        )
        joint_acceleration_cost = self._normalize_reward_value(
            self.acceleration_cost * adjusted_cost_weights["joint_acceleration"], 10.0
        )
        
        # 총 비용 합산
        costs = (flipped_cost + hip_ground_cost + shoulder_below_cost + 
                unwanted_contact_cost + front_contact_cost + front_feet_below_cost + joint_limit_cost +
                self_collision_cost + pitch_stability_cost + roll_stability_cost + 
                low_rear_hips_cost + body_height_cost + collision_cost +
                front_foot_height_cost + crossed_legs_cost + rear_feet_airborne_cost +
                vertical_vel_cost + orientation_cost +
                abduction_joints_cost + xy_angular_vel_cost + default_joint_cost +
                joint_velocity_cost + action_rate_cost + torque_cost + joint_acceleration_cost)
        
        # 비용 정보 저장 (음수로 저장)
        reward_info["flipped_over_cost"] = -flipped_cost
        reward_info["hip_ground_contact_cost"] = -hip_ground_cost
        reward_info["shoulder_below_pelvis_cost"] = -shoulder_below_cost
        reward_info["biped_front_contact_cost"] = -front_contact_cost
        reward_info["biped_unwanted_contact_cost"] = -unwanted_contact_cost
        reward_info["self_collision_cost"] = -self_collision_cost
        reward_info["reward_ctrl"] = -torque_cost
        
        # ==========================================
        # 최종 보상 계산 (범위: -10 ~ +10)
        # ==========================================
        raw_reward = rewards - costs
        
        # 이동 평균 기반 정규화 (선택적)
        if len(self._reward_history) > 10:
            self._reward_history.append(rewards)
            self._cost_history.append(costs)
            if len(self._reward_history) > self._reward_normalizer_window:
                self._reward_history.pop(0)
                self._cost_history.pop(0)
            
            # 안정적인 스케일링을 위한 percentile 기반 정규화
            reward_p95 = np.percentile(self._reward_history, 95)
            cost_p95 = np.percentile(self._cost_history, 95)
            
            if reward_p95 > 0 and cost_p95 > 0:
                normalized_rewards = rewards / reward_p95 * 5.0
                normalized_costs = costs / cost_p95 * 5.0
                raw_reward = normalized_rewards - normalized_costs
        
        # 최종 클리핑 및 커리큘럼 보너스
        final_reward = np.clip(raw_reward, -10.0, 10.0)
        final_reward = final_reward * (1.0 + 0.1 * self.curriculum_factor)  # 10% 보너스
        
        return final_reward, reward_info
    
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
        self._time_flipped_over = 0.0 #  타이머 초기화 추가
        self._time_shoulder_below_pelvis = 0.0  # 8. 어깨-골반 높이 타이머 초기화 
        self._time_hip_on_ground = 0.0  # 9. hip 접촉 타이머 초기화
        
        # ✨ [신규 추가] 실시간 저장을 위한 에피소드별 변수 초기화
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
    
    # ✨ [신규 추가] 실시간 저장을 위한 데이터 수집 메서드들
    def get_detailed_episode_info(self):
        """에피소드별 상세 정보를 반환합니다. 실시간 저장용입니다."""
        try:
            # 기본 에피소드 정보
            episode_info = {
                'episode_count': self._episode_count,
                'success_count': self._success_count,
                'success_rate': self._success_count / max(1, self._episode_count),
                'current_step': self._step,
                'max_episode_time': self._max_episode_time_sec,
                'time_remaining': max(0, self._max_episode_time_sec - self._step * self.dt),
            }
            
            # 로봇 상태 정보
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
            
            # 보상 및 비용 정보
            reward_info = {
                'current_episode_reward': getattr(self, '_current_episode_reward', 0.0),  # 현재 에피소드 누적 보상
                'current_episode_length': getattr(self, '_current_episode_length', 0),    # 현재 에피소드 길이
                'reward_weights': self.reward_weights.copy(),
                'cost_weights': self.cost_weights.copy(),
            }
            
            # 환경 설정 정보
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
            
            # 발 접촉 정보
            contact_info = {
                'feet_contact_forces': self.feet_contact_forces.tolist(),
                'front_feet_contact_forces': self.front_feet_contact_forces.tolist(),
                'feet_air_time': self._feet_air_time.tolist(),
                'last_contacts': self._last_contacts.tolist(),
            }
            
            # 안정성 메트릭
            stability_metrics = {
                'is_healthy': self.is_healthy,
                'is_flipped_over': self.is_flipped_over,
                'time_flipped_over': self._time_flipped_over,
                'is_shoulder_below_pelvis': self.is_shoulder_below_pelvis,
                'time_shoulder_below_pelvis': self._time_shoulder_below_pelvis,
                'is_hip_on_ground': self.is_hip_on_ground,
                'time_hip_on_ground': self._time_hip_on_ground,
                'shoulder_below_pelvis_cost': float(self.shoulder_below_pelvis_cost),
                'hip_ground_contact_cost': float(self.hip_ground_contact_cost),
                'trunk_up_alignment': float(self._trunk_up_alignment),
                'projected_gravity': self.projected_gravity.tolist(),
                'balance_history': self._balance_history.copy() if hasattr(self, '_balance_history') else [],
            }
            
            # 최근 액션 정보
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
            # 오류 발생 시 기본 정보만 반환
            return {
                'episode_info': {
                    'episode_count': self._episode_count,
                    'success_count': self._success_count,
                    'error': str(e)
                },
                'timestamp': time.time(),
                'error': f"상세 정보 수집 중 오류 발생: {str(e)}"
            }
    
    def get_environment_summary(self):
        """환경의 전체 요약 정보를 반환합니다. 실시간 저장용입니다."""
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
                'error': f"환경 요약 수집 중 오류 발생: {str(e)}",
                'timestamp': time.time(),
            }
    
    def get_performance_metrics(self):
        """현재 에피소드의 성능 메트릭을 반환합니다. 실시간 저장용입니다."""
        try:
            # 보상 컴포넌트 계산
            reward_components = {}
            if hasattr(self, 'reward_weights'):
                for key in self.reward_weights:
                    if hasattr(self, key):
                        try:
                            reward_components[key] = float(getattr(self, key))
                        except:
                            reward_components[key] = 0.0
            
            # 비용 컴포넌트 계산
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
            
            # 진행률
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
                'curriculum_factor': self.curriculum_factor,
                'timestamp': time.time(),
            }
        except Exception as e:
            return {
                'error': f"성능 메트릭 수집 중 오류 발생: {str(e)}",
                'timestamp': time.time(),
            }