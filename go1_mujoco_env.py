from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
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
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    BIPEDAL_READY_JOINTS = np.array([
        # 앞다리 (FR, FL) - 몸쪽으로 당긴 상태
        0.0, 4.0, -2.0,    # FR
        0.0, 4.0, -2.0,    # FL
        # 뒷다리 (RR, RL) - 더 안정적으로 웅크린 상태   
        0.0, 2.8, -1.2,    # RR
        0.0, 2.8, -1.2,    # RL
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
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0
        self._max_episode_time_sec = 120.0
        self._step = 0
        self._front_feet_touched = False

        # ✨ [신규 추가] 요청사항 반영: 회복 보상 및 불건강 상태 페널티 가중치
        self.reward_weights = {
            "linear_vel_tracking": 2.0,
            "angular_vel_tracking": 1.0,
            "healthy": 1.0,
            "feet_airtime": 5.0, 
            "recovery": 10.0,  # 회복 행동에 대한 보상 가중치
            "get_up": 20.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,
            "xy_angular_vel": 0.05,
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.0e-4,
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1,
            "unhealthy_state": 5.0, # 불건강 상태 지속에 대한 페널티 가중치
        }

        if self.biped:
            self.reward_weights["biped_upright"] = 15.0
            self.reward_weights["head_height"] = 10.0
            self.cost_weights["biped_front_contact"] = 50.0
            self.cost_weights["biped_rear_feet_airborne"] = 5.0
            self.cost_weights["biped_front_foot_height"] = 8.0
            self.cost_weights["biped_crossed_legs"] = 5.0
            self.cost_weights["biped_low_rear_hips"] = 9.0
            self.cost_weights["biped_front_feet_below_hips"] = 6.0
            self.cost_weights["biped_abduction_joints"] = 0.7
            self.cost_weights["biped_unwanted_contact"] = 150.0
            self.cost_weights["self_collision"] = 25.0
            self.cost_weights["head_low"] = 20.0  # 추가
            self.cost_weights["inverted_posture"] = 30.0  # 추가
            self.reward_weights["proper_orientation"] = 12.0  # 새로 추가

        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])
        
        self._desired_velocity_min = np.array([-0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25

        self._healthy_z_range = (0.22, 1.8)
        self._healthy_pitch_range = (-np.pi, 0.0)
        self._healthy_roll_range = (-np.deg2rad(80), np.deg2rad(80))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]
        self._cfrc_ext_front_feet_indices = [4, 7]
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]
        
        # ✨ [신규 추가] 요청사항 반영: 불건강 상태 지속 시간 관리 변수
        self._time_in_unhealthy_state = 0.0
        self._max_unhealthy_time = 1.0 # 0.5초 이상 지속 시 종료
        # ✨ [신규 추가] 요청사항 반영: 회복 보상 계산을 위한 이전 상태 저장 변수
        self._last_health_deviation = {"z": 0.0, "roll": 0.0, "pitch": 0.0}

        dof_position_limit_multiplier = 0.9
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1
        self._last_action = np.zeros(12)
        self._last_feet_contact_forces = np.zeros(4)
        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }
        self._head_site_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_SITE.value, "head"
        
        )
        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        # 힙(엉덩이) 바디 ID는 이족/사족 모드 모두에서 사용하므로 항상 준비합니다.
        hip_body_names = ["FR_hip", "FL_hip", "RR_hip", "RL_hip"]
        self._hip_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
            for name in hip_body_names
        ]

        if self.biped:
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
            self._rear_hips_min_height = 0.4
            
            front_hip_body_names = ["FR_hip", "FL_hip"]
            self._front_hip_body_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, name)
                for name in front_hip_body_names
            ]
            
            unwanted_contact_body_names = [
                "trunk",
                "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh",
                "FR_calf", "FL_calf",
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
    def proper_orientation_reward(self):
        """올바른 방향(머리가 위, 꼬리가 아래)으로 서있을 때 보상합니다."""
        # 머리와 꼬리(몸통 뒤쪽) 위치 비교
        head_pos = self.data.site_xpos[self._head_site_id]
        trunk_pos = self.data.xpos[self._main_body_id]
        
        # 머리의 X 좌표가 몸통보다 앞쪽(양수)에 있어야 함
        forward_alignment = head_pos[0] - trunk_pos[0]
        
        # 머리가 충분히 높고, 앞쪽에 있을 때 보상
        if head_pos[2] > trunk_pos[2] + 0.1 and forward_alignment > 0:
            height_bonus = (head_pos[2] - trunk_pos[2]) * 2.0
            forward_bonus = min(forward_alignment * 3.0, 1.0)
            return height_bonus + forward_bonus
        
        return 0.0

    @property
    def head_low_cost(self):
        """머리가 엉덩이보다 낮을 때 페널티를 부과합니다."""
        head_pos = self.data.site_xpos[self._head_site_id]
        # 뒷다리 엉덩이들의 평균 높이
        rear_hips_pos = self.data.xpos[self._rear_hip_body_ids]
        avg_hip_height = np.mean(rear_hips_pos[:, 2])
        
        # 머리가 엉덩이보다 낮으면 그 차이만큼 페널티
        if head_pos[2] < avg_hip_height:
            return (avg_hip_height - head_pos[2]) * 10.0
        return 0.0


    @property
    def inverted_posture_cost(self):
        """거꾸로 선 자세에 대한 페널티를 부과합니다."""
        # 몸통의 pitch 각도 확인
        state = self.state_vector()
        pitch = state[5]
        
        # pitch가 양수면 머리가 아래로 향함 (거꾸로 선 자세)
        if pitch > 0:
            return pitch * 20.0
        return 0.0

    @property
    def head_height_reward(self):
        """머리가 높은 위치에 있을 때 보상을 줍니다."""
        head_pos = self.data.site_xpos[self._head_site_id]
        
        # 머리 높이가 0.5m 이상일 때 보상
        if head_pos[2] > 0.5:
            return (head_pos[2] - 0.5) * 2.0
        return 0.0

    @property
    def get_up_reward(self):
        """[✨ 신규 추가] 넘어진 상태에서 일어서려는 행동을 직접적으로 보상합니다.

        이 보상은 두 가지 요소로 구성됩니다:
        1. 몸통의 높이(CoM 높이): 땅에서 몸을 밀어내 높이를 올릴수록 큰 보상을 받습니다.
        2. 몸통의 수평 유지: 몸통이 지면과 수평에 가까워질수록(뒤집히거나 옆으로 누운 상태에서 벗어날수록) 보상을 받습니다.
        
        이 함수는 로봇이 불건강(unhealthy) 상태일 때만 활성화됩니다.
        """
        is_ok, _, _ = self._get_health_status()
        if is_ok:
            return 0.0

        # 1. 몸통의 높이에 대한 보상 (0.0 ~ 0.22 사이의 높이를 0~1로 정규화)
        trunk_height = self.data.xpos[self._main_body_id][2]
        # 목표 높이(healthy_z_range의 최소값)에 가까워질수록 보상이 커집니다.
        height_reward = np.clip(trunk_height / self._healthy_z_range[0], 0.0, 1.0)

        # 2. 몸통의 수평 상태에 대한 보상 (non_flat_base_cost를 보상으로 전환)
        # projected_gravity의 x, y 성분이 0에 가까울수록(수평일수록) 보상이 커집니다.
        orientation_goodness = 1.0 - np.sum(np.square(self.projected_gravity[:2]))
        
        # 두 보상을 조합하여 반환합니다. 높이에 더 큰 가중치를 둡니다.
        return (height_reward * 1.5) + (orientation_goodness * 0.5)


    @property
    def acceleration_cost(self):
        """[✅ 수정] 실제 모터 특성을 반영하여 관절 가속도 페널티를 동적으로 조절합니다.

        관절 속도가 낮을 때 높은 가속이 발생하면 (짧고 빠른 진동) 더 큰 페널티를,
        속도가 높을 때 높은 가속이 발생하면 (움직임을 위한 자연스러운 가속) 더 작은 페널티를 부과합니다.
        이를 통해 불필요하게 빠른 발놀림을 줄이고 더 부드러운 움직임을 유도합니다.
        페널티는 (가속도^2) / (ㅣ속도ㅣ + ε) 에 비례하여 계산됩니다.
        """
        # 관절 속도의 절댓값과 관절 가속도를 가져옵니다.
        joint_velocities = np.abs(self.data.qvel[6:])
        joint_accelerations = self.data.qacc[6:]
        
        # 속도가 0에 가까울 때 분모가 0이 되는 것을 방지하기 위한 작은 값(epsilon)입니다.
        epsilon = 1e-6
        
        # 속도가 낮은 상태에서의 급가속에 더 큰 페널티를 부과하는 동적 페널티를 계산합니다.
        dynamic_penalty = np.sum(
            np.square(joint_accelerations) / (joint_velocities + epsilon)
        )
        
        return dynamic_penalty

    @property
    def self_collision_cost(self):
        """[✨ 신규 추가] 이족 보행 시, 팔과 다리의 자기-충돌에 대한 페널티를 계산합니다.
        
        모든 접촉점을 확인하여, (오른쪽 앞다리 - 왼쪽 앞다리) 또는 
        (오른쪽 뒷다리 - 왼쪽 뒷다리) 간의 충돌이 발생하면 페널티 카운트를 증가시킵니다.
        """
        cost = 0
        # 시뮬레이션의 모든 접촉(contact)을 순회합니다.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 접촉에 관여된 두 geom이 속한 body의 ID를 가져옵니다.
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]

            # 앞다리(팔) 간의 충돌 확인
            is_front_right_limb_contact = body1_id in self._front_right_limb_body_ids
            is_front_left_limb_contact = body2_id in self._front_left_limb_body_ids
            if is_front_right_limb_contact and is_front_left_limb_contact:
                cost += 1.0
                continue # 이미 충돌을 확인했으므로 다음 접촉으로 넘어갑니다.

            # (반대 순서로도 확인)
            is_front_right_limb_contact = body2_id in self._front_right_limb_body_ids
            is_front_left_limb_contact = body1_id in self._front_left_limb_body_ids
            if is_front_right_limb_contact and is_front_left_limb_contact:
                cost += 1.0
                continue

            # 뒷다리 간의 충돌 확인
            is_rear_right_limb_contact = body1_id in self._rear_right_limb_body_ids
            is_rear_left_limb_contact = body2_id in self._rear_left_limb_body_ids
            if is_rear_right_limb_contact and is_rear_left_limb_contact:
                cost += 1.0
                continue

            # (반대 순서로도 확인)
            is_rear_right_limb_contact = body2_id in self._rear_right_limb_body_ids
            is_rear_left_limb_contact = body1_id in self._rear_left_limb_body_ids
            if is_rear_right_limb_contact and is_rear_left_limb_contact:
                cost += 1.0

        return cost

    @property
    def biped_crossed_legs_cost(self):
        """[🚀 신규 추가] 이족 보행 시 뒷다리가 교차되는 것에 대한 페널티 함수입니다.
        
        오른쪽 뒷다리(RR_hip)의 Y좌표가 왼쪽 뒷다리(RL_hip)의 Y좌표보다 커지면
        (즉, 다리가 꼬이면) 그 차이만큼 페널티를 부과합니다.
        로봇이 정면을 바라볼 때, RR_hip의 Y좌표는 RL_hip의 Y좌표보다 작아야 정상입니다.
        """
        rear_hips_pos = self.data.xpos[self._rear_hip_body_ids]
        
        # rear_hips_pos[0]은 RR_hip, rear_hips_pos[1]은 RL_hip의 좌표입니다.
        y_rr = rear_hips_pos[0, 1]
        y_rl = rear_hips_pos[1, 1]
        
        # y_rr이 y_rl보다 클 때만 페널티를 계산합니다.
        cost = max(0, y_rr - y_rl)
        
        return cost

    @property
    def biped_low_rear_hips_cost(self):
        """[🚀 신규 추가 & ✅ 수정] 이족 보행 시 뒷다리 고관절이 너무 낮아지는 것에 대한 페널티 함수입니다.
        
        각 뒷다리 고관절의 Z좌표가 미리 정의된 최소 높이(_rear_hips_min_height)보다
        낮아질 경우, 그 차이만큼 페널티를 부과합니다. 이 기준값은 땅에 거의 닿는 수준으로 설정됩니다.
        """
        rear_hips_pos = self.data.xpos[self._rear_hip_body_ids]
        
        # Z 좌표(높이)만 추출합니다.
        hips_z = rear_hips_pos[:, 2]
        
        # 최소 높이에서 현재 높이를 뺍니다. 이 값이 양수이면 기준보다 낮은 것입니다.
        height_difference = self._rear_hips_min_height - hips_z
        
        # 기준보다 높은 경우(음수 값)는 페널티가 없도록 0으로 만듭니다.
        cost = np.sum(height_difference.clip(min=0.0))
        
        return cost*10.0

    @property
    def biped_front_feet_below_hips_cost(self):
        """[✅ 추가] 앞발이 앞쪽 고관절보다 낮아지는 것에 대한 페널티 함수입니다.
        
        앞발(site)의 Z좌표가 앞쪽 고관절(hip body)의 Z좌표보다 낮을 경우,
        그 차이의 제곱만큼 페널티를 부과하여 더 강력하게 제지합니다.
        """
        front_feet_pos = self.data.site_xpos[self._front_feet_site_ids]
        front_hips_pos = self.data.xpos[self._front_hip_body_ids]

        # Z 좌표(높이)만 추출합니다.
        feet_z = front_feet_pos[:, 2]
        hips_z = front_hips_pos[:, 2]

        # 고관절 높이에서 발 높이를 뺍니다. 이 값이 양수이면 발이 더 낮은 것입니다.
        height_difference = hips_z - feet_z
        
        # 발이 고관절보다 높은 경우(음수 값)는 페널티가 없도록 0으로 만듭니다.
        # 차이의 제곱을 사용하여 더 낮은 위치에 대해 더 큰 페널티를 부과합니다.
        cost = np.sum(np.square(height_difference.clip(min=0.0)))
        
        return cost

    @property
    def trunk_forward_axis_in_world(self):
        """[💡 추가] 몸통의 전방(X) 축 벡터를 월드 좌표계 기준으로 반환합니다."""
        return self.data.xmat[self._main_body_id].reshape(3, 3)[:, 0]

    @property
    def front_feet_contact_forces(self):
        """Returns the contact forces on the front feet."""
        front_feet_forces = self.data.cfrc_ext[self._cfrc_ext_front_feet_indices]
        return np.linalg.norm(front_feet_forces, axis=1)

    @property
    def biped_upright_reward(self):
        """이족 보행 시 몸통을 수직으로 유지하는 것에 대한 보상 함수입니다."""
        world_up_vector = np.array([0, 0, 1])
        trunk_forward_vector = self.trunk_forward_axis_in_world
        
        # 머리 방향 확인 - 머리가 아래를 향하면 페널티
        head_pos = self.data.site_xpos[self._head_site_id]
        trunk_pos = self.data.xpos[self._main_body_id]
        
        # 머리가 몸통보다 낮으면 보상을 0으로
        if head_pos[2] < trunk_pos[2]:
            return 0.0
        
        # 몸통의 전방 축이 위를 향해야 함
        alignment = np.dot(trunk_forward_vector, world_up_vector)
        
        # alignment가 0.7 이상일 때만 보상 (약 45도 이상 서있을 때)
        if alignment < 0.7:
            return 0.0
        
        return alignment

    @property
    def biped_front_foot_height_cost(self):
        """[💡 추가] 앞발이 무릎보다 낮아지는 것에 대한 페널티 함수입니다.
        
        앞발(site)의 Z좌표가 앞쪽 무릎(calf body)의 Z좌표보다 낮을 경우,
        그 차이만큼 페널티를 부과합니다.
        """
        front_feet_pos = self.data.site_xpos[self._front_feet_site_ids]
        front_knees_pos = self.data.xpos[self._front_knee_body_ids]

        # Z 좌표(높이)만 추출합니다.
        feet_z = front_feet_pos[:, 2]
        knees_z = front_knees_pos[:, 2]

        # 무릎 높이에서 발 높이를 뺍니다. 이 값이 양수이면 발이 더 낮은 것입니다.
        height_difference = knees_z - feet_z
        
        # 발이 무릎보다 높은 경우(음수 값)는 페널티가 없도록 0으로 만듭니다.
        cost = np.sum(height_difference.clip(min=0.0))
        
        return cost


    @property
    def biped_front_contact_cost(self):
        """Penalizes contact on the front feet."""
        contact_forces = self.front_feet_contact_forces
        # Penalize any contact force on the front feet using its squared magnitude
        return np.sum(np.square(contact_forces))


    def _get_health_status(self):
        """로봇의 건강 상태를 확인하고, 종료 시 원인과 상세 정보를 반환합니다."""
        state = self.state_vector()

        # 상태 유효성 검사
        if not np.isfinite(state).all():
            details = f"State values are not finite: {state}"
            return False, "state_not_finite", details

        # Z축 높이 검사
        min_z, max_z = self._healthy_z_range
        if not (min_z <= state[2] <= max_z):
            details = f"Z-position: {state[2]:.3f}, Healthy Range: [{min_z:.2f}, {max_z:.2f}]"
            return False, "unhealthy_z", details

        # Roll 각도 검사
        min_roll, max_roll = self._healthy_roll_range
        if not (min_roll <= state[4] <= max_roll):
            details = f"Roll: {state[4]:.3f} rad, Healthy Range: [{min_roll:.2f}, {max_roll:.2f}] rad"
            return False, "unhealthy_roll", details

        # Pitch 각도 검사
        min_pitch, max_pitch = self._healthy_pitch_range
        if not (min_pitch <= state[5] <= max_pitch):
            details = f"Pitch: {state[5]:.3f} rad, Healthy Range: [{min_pitch:.2f}, {max_pitch:.2f}] rad"
            return False, "unhealthy_pitch", details

        # 엉덩이(힙) 바디가 바닥(월드 바디: id=0)과 접촉하면 unhealthy로 판정합니다.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1_id = self.model.geom_bodyid[contact.geom1]
            body2_id = self.model.geom_bodyid[contact.geom2]
            if (body1_id in self._hip_body_ids and body2_id == 0) or (
                body2_id in self._hip_body_ids and body1_id == 0
            ):
                details = "Hip body contacted the ground (world body)"
                return False, "hip_contact_with_ground", details

        # ✨ [수정] 이족 보행 시 앞발 접촉 종료 조건을 삭제합니다.
        # 이 조건은 이제 _calc_reward 함수에서 패널티로만 처리됩니다.
        if self.biped:
            pass
            # 앞발 접촉 검사
            # if np.any(self.front_feet_contact_forces > 1.0):
            #     forces = self.front_feet_contact_forces
            #     details = f"Front feet contact forces: [FR={forces[0]:.2f}, FL={forces[1]:.2f}], Threshold: > 1.0"
            #     return False, "front_foot_contact", details

        # 모든 검사를 통과한 경우
        return True, "not_terminated", "No termination"

    def step(self, action):
        self._step += 1
        
        # biped 모드에서 앞발이 닿았는지 여부를 기록하는 로직은 유지합니다.
        if self.biped:
            if np.any(self.front_feet_contact_forces > 1.0):
                self._front_feet_touched = True

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        
        # --- [✅ 수정] 종료 조건 로직을 완전히 변경합니다. ---
        is_currently_healthy, reason, details = self._get_health_status()

        if not is_currently_healthy:
            # 불건강 상태가 지속되면 타이머를 증가시킵니다.
            if reason != "unhealthy_z":
                self._time_in_unhealthy_state += self.dt
            else:
                self._time_in_unhealthy_state += self.dt
        else:
            # 건강한 상태로 돌아오면 타이머를 리셋합니다.
            self._time_in_unhealthy_state = 0.0

        # 불건강 상태가 _max_unhealthy_time(15초) 이상 지속되면 에피소드를 종료합니다.
        terminated = self._time_in_unhealthy_state > self._max_unhealthy_time
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
            # 타이머에 의해 종료된 경우, 마지막으로 감지된 불건강 원인을 기록합니다.
            info["termination_reason"] = f"prolonged_{reason}"
            info["termination_details"] = f"{details} (persisted for > {self._max_unhealthy_time:.2f}s)"
        elif not is_currently_healthy:
            # 종료되지는 않았지만 현재 불건강 상태인 경우, 그 상태를 기록하여 디버깅을 돕습니다.
            info["termination_reason"] = "unhealthy_state_active"
            info["termination_details"] = details
        # --- 로직 변경 끝 ---

        if truncated and self.biped and not self._front_feet_touched:
            info["bipedal_success"] = True

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action
        self._last_feet_contact_forces = self.feet_contact_forces.copy()

        return observation, reward, terminated, truncated, info

    @property
    def recovery_reward(self):
        """[✨ 신규 추가] 넘어진 상태에서 다시 일어나려는 행동에 보상을 줍니다.

        로봇이 불건강(unhealthy) 상태일 때, 건강한 상태(정상 높이 및 각도)에
        얼마나 가까워졌는지를 측정하여 보상을 계산합니다.
        이전 스텝보다 건강한 범위에 가까워질수록 양의 보상을 받습니다.
        """
        is_ok, _, _ = self._get_health_status()
        
        state = self.state_vector()
        z, roll, pitch = state[2], state[4], state[5]

        # 현재 상태가 건강한 범위에서 얼마나 벗어났는지(편차) 계산합니다.
        z_dev = 0
        if not (self._healthy_z_range[0] <= z <= self._healthy_z_range[1]):
            z_dev = min(abs(z - self._healthy_z_range[0]), abs(z - self._healthy_z_range[1]))

        roll_dev = 0
        if not (self._healthy_roll_range[0] <= roll <= self._healthy_roll_range[1]):
            roll_dev = min(abs(roll - self._healthy_roll_range[0]), abs(roll - self._healthy_roll_range[1]))

        pitch_dev = 0
        if not (self._healthy_pitch_range[0] <= pitch <= self._healthy_pitch_range[1]):
            pitch_dev = min(abs(pitch - self._healthy_pitch_range[0]), abs(pitch - self._healthy_pitch_range[1]))
        
        current_deviation = {
            "z": z_dev,
            "roll": roll_dev,
            "pitch": pitch_dev
        }
        
        # 건강하다면 회복 보상은 없으며, 다음 계산을 위해 이전 편차를 0으로 초기화합니다.
        if is_ok:
            self._last_health_deviation = {"z": 0.0, "roll": 0.0, "pitch": 0.0}
            return 0.0

        # 이전 스텝 대비 편차가 얼마나 '감소'했는지(개선되었는지) 계산합니다.
        z_improvement = self._last_health_deviation["z"] - current_deviation["z"]
        roll_improvement = self._last_health_deviation["roll"] - current_deviation["roll"]
        pitch_improvement = self._last_health_deviation["pitch"] - current_deviation["pitch"]
        
        # 다음 스텝에서 사용할 수 있도록 현재 편차를 저장합니다.
        self._last_health_deviation = current_deviation

        # 개선 정도의 합을 보상으로 반환합니다. 값이 양수이면 자세가 나아졌다는 의미입니다.
        return z_improvement + roll_improvement + pitch_improvement

    @property
    def unhealthy_state_cost(self):
        """[✨ 신규 추가] 불건강 상태에 대한 페널티를 계산합니다.
        
        로봇이 건강하지 않은 상태(넘어져 있는 등)에 머무를 경우 1.0의 비용을 반환합니다.
        이를 통해 에이전트는 불건강한 상태를 피하도록 학습합니다.
        """
        is_ok, _, _ = self._get_health_status()
        return 0.0 if is_ok else 1.0
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

    ######### Positive Reward functions #########
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
    def heading_tracking_reward(self):
        # TODO: qpos[3:7] are the quaternion values
        pass

    @property
    def feet_air_time_reward(self):
        """[✅ 수정] 큰 걸음을 유도하기 위해, 발이 공중에 머무는 시간의 '제곱'에 비례한 보상을 줍니다.
        
        기존의 선형적인 보상 방식(air_time - 0.2) 대신, air_time의 제곱을 사용합니다.
        이를 통해 공중에 약간 더 길게 머무르는 행동에 훨씬 더 큰 보상을 부여하여, 
        에이전트가 더 동적이고 큰 보폭을 취하도록 강력하게 유도합니다.
        최소 시간(0.2초)을 넘겼을 때만 보상을 주는 조건은 유지합니다.
        """
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        
        # 이족 모드에서는 뒷발(RR, RL)만 보상에 포함하고, 앞발(FR, FL)은 제외합니다.
        # 비-이족 모드에서는 모든 발을 동일하게 보상에 포함합니다.
        if self.biped:
            reward_feet_mask = np.array([0.0, 0.0, 1.0, 1.0])
        else:
            reward_feet_mask = np.ones(4)

        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # ✅ [수정] 보상 계산 방식을 선형에서 제곱으로 변경
        # air_time이 0.2를 넘는 구간에 대해서 제곱의 보상을 줍니다.
        time_since_threshold = (self._feet_air_time - 0.2).clip(min=0.0)
        air_time_reward = np.sum(
            np.square(time_since_threshold) * first_contact * reward_feet_mask
        )
        
        # 목표 속도가 매우 낮을 때는 보상을 주지 않는 조건 (기존과 동일)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        is_ok, _, _ = self._get_health_status()
        return 1.0 if is_ok else 0.0

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.projected_gravity[:2]))

    @property
    def collision_cost(self):
        # Penalize collisions on selected bodies
        return np.sum(
            1.0
            * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
        )

    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # Last 12 values are the motor torques
        torque_val = np.sum(np.square(self.data.qfrc_actuator[-12:]))

        # ✨ [수정] 불건강 상태일 때 페널티를 90% 할인하여 과감한 행동을 유도합니다.
        is_ok, _, _ = self._get_health_status()
        if not is_ok:
            return torque_val * 0.1
        return torque_val

    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        action_rate_val = np.sum(np.square(self._last_action - action))

        # ✨ [수정] 불건강 상태일 때 페널티를 90% 할인하여 과감한 행동을 유도합니다.
        is_ok, _, _ = self._get_health_status()
        if not is_ok:
            return action_rate_val * 0.1
        return action_rate_val

    @property
    def joint_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[6:]))

    @property
    def acceleration_cost(self):
        """[✅ 수정] 실제 모터 특성을 반영하여 관절 가속도 페널티를 동적으로 조절합니다.
        ... (기존 주석) ...
        """
        joint_velocities = np.abs(self.data.qvel[6:])
        joint_accelerations = self.data.qacc[6:]
        epsilon = 1e-6
        dynamic_penalty = np.sum(
            np.square(joint_accelerations) / (joint_velocities + epsilon)
        )
        
        # ✨ [수정] 불건강 상태일 때 페널티를 90% 할인하여 과감한 행동을 유도합니다.
        is_ok, _, _ = self._get_health_status()
        if not is_ok:
            return dynamic_penalty * 0.1
        return dynamic_penalty

    @property
    def default_joint_position_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))


    @property
    def biped_abduction_joints_cost(self):
        """[✨ 신규 추가] 이족 보행 시 어깨/엉덩이 관절(abduction)이 0에 가깝도록 유도하는 페널티입니다.
        
        관련 관절 각도의 제곱 합을 계산하여, 0에서 벗어날수록 더 큰 페널티를 부과합니다.
        (Indices: 0=FR_hip, 3=FL_hip, 6=RR_hip, 9=RL_hip)
        """
        abduction_joints_indices = [0, 3, 6, 9]
        dofs_position = self.data.qpos[7:]
        abduction_angles = dofs_position[abduction_joints_indices]
        
        return np.sum(np.square(abduction_angles))

    @property
    def biped_unwanted_contact_cost(self):
        """[✨ 신규 추가] 이족 보행 시, 뒷발을 제외한 신체 부위의 접촉에 대해 큰 페널티를 부과합니다.
        
        몸통(trunk), 모든 허벅지(thighs), 앞쪽 종아리(calves)의 접촉 힘을 확인하고,
        접촉이 발생하면 힘의 제곱에 비례하는 페널티를 적용합니다.
        """
        contact_forces = self.data.cfrc_ext[self._unwanted_contact_body_ids]
        # 각 부위별 접촉 힘의 크기(norm)를 계산하고, 그 값의 제곱 합을 페널티로 사용합니다.
        cost = np.sum(np.square(np.linalg.norm(contact_forces, axis=1)))
        return cost

    @property
    def smoothness_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._last_action))

    @property
    def curriculum_factor(self):
        return self._curriculum_base**0.997

    def _calc_reward(self, action):
        # Positive Rewards
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        
        # ✨ [신규 추가] 회복 보상을 전체 보상에 추가합니다.
        recovery_reward = self.recovery_reward * self.reward_weights["recovery"]
        # ✨ [신규 추가] 일어서기 보상을 전체 보상에 추가합니다.
        get_up_reward = self.get_up_reward * self.reward_weights["get_up"]

        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
            + recovery_reward
            + get_up_reward # ✨ 추가
        )

        # Negative Costs
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )
        
        # ✨ [신규 추가] 불건강 상태 비용을 전체 비용에 추가합니다.
        unhealthy_state_cost = self.unhealthy_state_cost * self.cost_weights["unhealthy_state"]

        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_velocity_cost
            + joint_acceleration_cost
            + collision_cost
            + unhealthy_state_cost
        )

        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "recovery_reward": recovery_reward,
            "get_up_reward": get_up_reward, # ✨ 로그 추가
            "unhealthy_state_cost": -unhealthy_state_cost,
        }

        if self.biped:
            upright_reward = self.biped_upright_reward * self.reward_weights["biped_upright"]
            head_height_reward = self.head_height_reward * self.reward_weights["head_height"]  # 추가
            proper_orientation_reward = self.proper_orientation_reward * self.reward_weights["proper_orientation"]  # 새로 추가
            front_contact_cost = self.biped_front_contact_cost * self.cost_weights["biped_front_contact"]
            front_foot_height_cost = self.biped_front_foot_height_cost * self.cost_weights["biped_front_foot_height"]
            crossed_legs_cost = self.biped_crossed_legs_cost * self.cost_weights["biped_crossed_legs"]
            low_rear_hips_cost = self.biped_low_rear_hips_cost * self.cost_weights["biped_low_rear_hips"]
            front_feet_below_hips_cost = self.biped_front_feet_below_hips_cost * self.cost_weights["biped_front_feet_below_hips"]
            abduction_joints_cost = self.biped_abduction_joints_cost * self.cost_weights["biped_abduction_joints"]
            unwanted_contact_cost = self.biped_unwanted_contact_cost * self.cost_weights["biped_unwanted_contact"]
            self_collision_cost_val = self.self_collision_cost * self.cost_weights["self_collision"]
            head_low_cost = self.head_low_cost * self.cost_weights["head_low"]  # 추가
            inverted_posture_cost = self.inverted_posture_cost * self.cost_weights["inverted_posture"]  # 추가
            

            rear_feet_airborne_cost = 0.0
            if np.all(self.feet_contact_forces[2:] < 1.0):
                rear_feet_airborne_cost = self.cost_weights["biped_rear_feet_airborne"]

            rewards += upright_reward
            rewards += proper_orientation_reward  # 새로 추가
            rewards += head_height_reward  # 추가
            costs += head_low_cost  # 추가
            costs += inverted_posture_cost  # 추가
            costs += front_contact_cost
            costs += rear_feet_airborne_cost
            costs += front_foot_height_cost
            costs += crossed_legs_cost
            costs += low_rear_hips_cost
            costs += front_feet_below_hips_cost
            costs += abduction_joints_cost
            costs += unwanted_contact_cost
            costs += self_collision_cost_val

            reward_info["biped_upright_reward"] = upright_reward
            reward_info["biped_front_contact_cost"] = -front_contact_cost
            reward_info["proper_orientation_reward"] = proper_orientation_reward  # 새로 추가
            reward_info["biped_rear_feet_airborne_cost"] = -rear_feet_airborne_cost
            reward_info["biped_front_foot_height_cost"] = -front_foot_height_cost
            reward_info["biped_crossed_legs_cost"] = -crossed_legs_cost
            reward_info["biped_low_rear_hips_cost"] = -low_rear_hips_cost
            reward_info["biped_front_feet_below_hips_cost"] = -front_feet_below_hips_cost
            reward_info["biped_abduction_joints_cost"] = -abduction_joints_cost
            reward_info["biped_unwanted_contact_cost"] = -unwanted_contact_cost
            reward_info["self_collision_cost"] = -self_collision_cost_val
            reward_info["head_height_reward"] = head_height_reward
            reward_info["head_low_cost"] = -head_low_cost
            reward_info["inverted_posture_cost"] = -inverted_posture_cost
        else:
            costs += orientation_cost
            costs += default_joint_position_cost
            reward_info["orientation_cost"] = -orientation_cost
            reward_info["default_joint_position_cost"] = -default_joint_position_cost

        reward = max(0.0, rewards - costs)

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The above seven values are ignored since they are privileged information
        # The remaining 12 values are the joint positions
        # The joint positions are relative to the starting position
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    def reset_model(self):
        qpos = self.model.key_qpos[0].copy()

        # ✨ [수정] 20% 확률로 넘어진 상태에서 시작하는 커리큘럼 학습 적용
        if False:#np.random.rand() < 0.000:
            # 옆으로 또는 뒤로 누운 자세를 만듭니다.
            # Roll 또는 Pitch 각도를 크게 주어 눕힙니다.
            random_angle = np.random.uniform(np.pi / 2.1, np.pi / 1.5) # 85~120도 사이
            
            # Roll(옆으로) 또는 Pitch(앞뒤로) 중 무작위로 선택
            if np.random.rand() < 0.5: # Roll
                rot_quat = np.array([np.cos(random_angle / 2), np.sin(random_angle / 2), 0, 0])
            else: # Pitch
                rot_quat = np.array([np.cos(random_angle / 2), 0, np.sin(random_angle / 2), 0])

            qpos[3:7] = rot_quat
            qpos[2] = 0.1 # 높이를 낮게 설정
        
        elif self.biped:
            qpos[7:] = self.BIPEDAL_READY_JOINTS
            qpos[2] = 0.65
            # pitch를 -95도에서 -85도로 수정 (더 수직에 가깝게)
            pitch_angle = np.deg2rad(-85)  
            pitch_quaternion = np.array([np.cos(pitch_angle / 2), 0, np.sin(pitch_angle / 2), 0])
            qpos[3:7] = pitch_quaternion

        if self._rand_power > 0.0:
            joint_noise = np.random.normal(
                loc=0.0,
                scale=0.1 * self._rand_power,
                size=qpos[7:].shape
            )
            qpos[7:] += joint_noise
            joint_limits = self.model.jnt_range[1:, :]
            qpos[7:] = np.clip(qpos[7:], joint_limits[:, 0], joint_limits[:, 1])

        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qpos[7:].copy()

        self._desired_velocity = self._sample_desired_vel()

        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self._front_feet_touched = False
        self._last_feet_contact_forces = np.zeros(4)
        
        self._time_in_unhealthy_state = 0.0
        self._last_health_deviation = {"z": 0.0, "roll": 0.0, "pitch": 0.0}

        observation = self._get_obs()
        return observation


    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
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

        return roll_x, pitch_y, yaw_z  # in radians