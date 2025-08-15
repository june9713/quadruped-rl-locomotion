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

    def __init__(self, ctrl_type="torque", biped=False, **kwargs):
        model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml")
        self.biped = biped

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  # Manually set afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0
        self._max_episode_time_sec = 15.0
        self._step = 0
        self._front_feet_touched = False

        # Weights for the reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 2.0,  # Was 1.0
            "angular_vel_tracking": 1.0,
            "healthy": 0.0,  # was 0.05
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0,  # Was 1.0
            "xy_angular_vel": 0.05,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7, 
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        if self.biped:
            # Add new weights for bipedal walking
            self.reward_weights["biped_upright"] = 5.0
            self.cost_weights["biped_front_contact"] = 10.0
            # ✨ 추가: 뒷다리가 동시에 공중에 뜨는 것에 대한 패널티 가중치
            self.cost_weights["biped_rear_feet_airborne"] = 5.0

        self._curriculum_base = 0.3
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])

        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()  # [0.5, 0.0, 0.0]
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.22, 1.8)
        self._healthy_pitch_range = (-np.deg2rad(175), np.deg2rad(-10)) 
        self._healthy_roll_range = (-np.deg2rad(80), np.deg2rad(80))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        self._cfrc_ext_front_feet_indices = [4, 7] # FR, FL
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        # ✨ 추가: 이전 스텝의 발 접촉력을 저장할 변수
        self._last_feet_contact_forces = np.zeros(4)

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        # Feet site names to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
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

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

    @property
    def front_feet_contact_forces(self):
        """Returns the contact forces on the front feet."""
        front_feet_forces = self.data.cfrc_ext[self._cfrc_ext_front_feet_indices]
        return np.linalg.norm(front_feet_forces, axis=1)

    @property
    def biped_upright_reward(self):
        """Reward for keeping the trunk upright (roll and pitch close to zero)."""
        # The cost for non-flat base is the sum of squares of the xy components of projected gravity
        # We can convert this to a reward.
        orientation_error = np.sum(np.square(self.projected_gravity[:2]))
        # Use an exponential reward to give high reward for being very upright.
        return np.exp(-5.0 * orientation_error) 

    @property
    def biped_front_contact_cost(self):
        """Penalizes contact on the front feet."""
        contact_forces = self.front_feet_contact_forces
        # Penalize any contact force on the front feet using its squared magnitude
        return np.sum(np.square(contact_forces))


    def _check_health(self):
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

        # 이족 보행 추가 조건 검사
        if self.biped:
            # 앞발 접촉 검사
            if np.any(self.front_feet_contact_forces > 1.0):
                forces = self.front_feet_contact_forces
                details = f"Front feet contact forces: [FR={forces[0]:.2f}, FL={forces[1]:.2f}], Threshold: > 1.0"
                return False, "front_foot_contact", details

            # ✨ [수정] 뒷발 공중 검사 로직을 여기서 삭제합니다.
            # 이 로직은 이제 _calc_reward 함수에서 패널티로 처리됩니다.
            # if np.all(self.feet_contact_forces[2:] < 1.0):
            #     prev_forces = self._last_feet_contact_forces[2:]
            #     details = f"Forces in prev_step: [RR={prev_forces[0]:.2f}, RL={prev_forces[1]:.2f}], Condition: Both forces dropped < 1.0"
            #     return False, "biped_rear_feet_in_air", details

        # 모든 검사를 통과한 경우
        return True, "not_terminated", "No termination"

    def step(self, action):
        self._step += 1
        
        # ✨ Note: front_contact_in_step 변수는 이제 _check_health에서 처리하므로 삭제해도 무방하나,
        # 다른 로직(self._front_feet_touched)에 사용되므로 유지합니다.
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

        # ✨ 수정된 부분: 복잡한 if/elif 블록 대신 _check_health 함수를 직접 호출
        if terminated:
            is_ok, reason, details = self._check_health()
            if not is_ok: # health check가 False를 반환했을 경우
                info["termination_reason"] = reason
                info["termination_details"] = details
            else: # 드물지만 is_healthy와 _check_health 사이에 불일치가 발생할 경우를 대비한 방어 코드
                info["termination_reason"] = "unknown_cause_logic_error"
                info["termination_details"] = "is_healthy was False, but _check_health returned True."

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
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0

        # --- 제안 3: 이족 보행 시 교차 보상 추가 ---
        if self.biped:
            # 뒷다리(RR, RL)의 접촉 상태만 사용
            rear_feet_contact = curr_contact[2:]
            # 한 발은 닿고, 다른 한 발은 떨어져 있을 때 1.0의 보상
            is_alternating = (rear_feet_contact[0] != rear_feet_contact[1])
            return float(is_alternating)
        # --- 제안 3 끝 ---

        # (기존 네 발 보행 로직은 그대로 유지)
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        air_time_reward = np.sum((self._feet_air_time - 0.2) * first_contact)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

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
    def acceleration_cost(self):
        return np.sum(np.square(self.data.qacc[6:]))

    @property
    def default_joint_position_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))

    @property
    def smoothness_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._last_action))

    @property
    def curriculum_factor(self):
        return self._curriculum_base**0.997

    def _calc_reward(self, action):
        # TODO: Add debug mode with custom Tensorboard calls for individual reward
        #   functions to get a better sense of the contribution of each reward function
        # TODO: Cost for thigh or calf contact with the ground

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
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
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

        # 🐞 [버그 수정] 누락되었던 collision_cost를 비용 합산에 추가합니다.
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            + default_joint_position_cost
            + collision_cost  # ✨ collision_cost 추가
        )

        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        if self.biped:
            upright_reward = self.biped_upright_reward * self.reward_weights["biped_upright"]
            front_contact_cost = self.biped_front_contact_cost * self.cost_weights["biped_front_contact"]

            # ✨ 추가: 뒷다리 동시 공중 부양 패널티 계산
            rear_feet_airborne_cost = 0.0
            if np.all(self.feet_contact_forces[2:] < 1.0):
                rear_feet_airborne_cost = self.cost_weights["biped_rear_feet_airborne"]

            rewards += upright_reward
            costs += front_contact_cost
            costs += rear_feet_airborne_cost # ✨ 계산된 패널티 추가

            reward_info["biped_upright_reward"] = upright_reward
            reward_info["biped_front_contact_cost"] = -front_contact_cost
            reward_info["biped_rear_feet_airborne_cost"] = -rear_feet_airborne_cost # ✨ 정보 로깅

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
        # XML의 기본 'home' 자세(key_qpos[0])를 기반으로 시작합니다.
        qpos = self.model.key_qpos[0].copy()

        if self.biped:
            # 1. 위에서 수정한 안정적인 관절 각도로 설정합니다.
            qpos[7:] = self.BIPEDAL_READY_JOINTS

            # 2. 몸통의 초기 높이를 안전하게 설정합니다.
            qpos[2] = 0.65

            # 3. 몸통을 수직으로 세우는 쿼터니언 값을 계산하여 설정합니다.
            pitch_angle = -np.pi / 2  # -90 degrees in radians
            pitch_quaternion = np.array([np.cos(pitch_angle / 2), 0, np.sin(pitch_angle / 2), 0])
            qpos[3:7] = pitch_quaternion

        # ✨ 수정된 부분: 랜덤 노이즈 없이 계산된 자세를 그대로 적용합니다.
        self.data.qpos[:] = qpos

        # ✨ 수정된 부분: 초기 제어(ctrl) 값도 랜덤 노이즈 없이 설정합니다.
        if self.biped:
            # position control 모드를 가정하고, 목표 관절 각도를 초기 ctrl 값으로 설정
            self.data.ctrl[:] = self.BIPEDAL_READY_JOINTS.copy()
        else:
            self.data.ctrl[:] = self.model.key_ctrl[0].copy()

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0
        self._front_feet_touched = False
        
        self._last_feet_contact_forces = np.zeros(4)

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