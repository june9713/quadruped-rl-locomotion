#!/usr/bin/env python3
"""
Go1 2족 보행 (Standing/Bipedal Walking) 환경
"""

import numpy as np
import mujoco
from go1_mujoco_env import Go1MujocoEnv
import math
from collections import deque

# visual_train.py에서 import할 수 있도록 환경 이름 추가
__all__ = ['Go1StandingEnv', 'GradualStandingEnv', 'StandingReward']


class StandingReward:
    """2족 보행을 위한 보상 함수"""

    def __init__(self):
        # 보상 가중치들
        self.weights = {
            'upright': 15.0,        # 똑바로 서있기
            'height': 10.0,         # 적절한 높이 유지
            'balance': 8.0,         # 균형 유지
            'foot_contact': 5.0,    # 뒷발로만 서있기
            'forward_vel': 3.0,     # 전진 속도
            'lateral_stability': 4.0, # 좌우 안정성
            'energy': -0.05,        # 에너지 효율
            'joint_limit': -5.0,    # 관절 한계 페널티
            'symmetry': 3.0,        # 좌우 대칭성
            'smooth_motion': 2.0    # 부드러운 동작
        }

    def compute_reward(self, model, data):
        """2족 보행 보상 계산"""
        total_reward = 0.0
        reward_info = {}

        # 1. 똑바로 서있기 보상
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]  # z축 방향

        # 몸체가 똑바로 서있어야 함
        upright_reward = up_vector[2]  # z 성분이 1에 가까울수록 좋음
        upright_reward = max(0, upright_reward)
        total_reward += self.weights['upright'] * upright_reward
        reward_info['upright'] = upright_reward

        # 2. 높이 보상
        trunk_height = data.qpos[2]
        target_height = 0.5  # 2족 보행시 목표 높이
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-10 * height_error)
        total_reward += self.weights['height'] * height_reward
        reward_info['height'] = height_reward

        # 3. 발 접촉 보상 (뒷발로만 서있어야 함)
        foot_contacts = self._get_foot_contacts(model, data)
        # 앞발(FR, FL)은 땅에 닿으면 안됨, 뒷발(RR, RL)은 닿아야 함
        front_foot_penalty = foot_contacts[0] + foot_contacts[1]  # FR + FL
        rear_foot_reward = foot_contacts[2] + foot_contacts[3]    # RR + RL

        foot_reward = rear_foot_reward - 2 * front_foot_penalty
        total_reward += self.weights['foot_contact'] * foot_reward
        reward_info['foot_contact'] = foot_reward

        # 4. 균형 보상
        trunk_vel = data.qvel[:3]
        trunk_angular_vel = data.qvel[3:6]

        # 너무 빠르게 움직이지 않아야 함
        linear_stability = np.exp(-2 * np.linalg.norm(trunk_vel[1:]))  # y,z 속도 제한
        angular_stability = np.exp(-3 * np.linalg.norm(trunk_angular_vel))

        balance_reward = linear_stability * angular_stability
        total_reward += self.weights['balance'] * balance_reward
        reward_info['balance'] = balance_reward

        # 5. 전진 속도 보상 (적당한 속도로 전진)
        forward_vel = trunk_vel[0]  # x 방향 속도
        target_vel = 0.3  # 목표 전진 속도

        # 너무 빠르거나 뒤로 가면 페널티
        if forward_vel < 0:
            vel_reward = 0
        elif forward_vel > target_vel * 2:
            vel_reward = np.exp(-5 * (forward_vel - target_vel * 2))
        else:
            vel_reward = forward_vel / target_vel

        total_reward += self.weights['forward_vel'] * vel_reward
        reward_info['forward_vel'] = vel_reward

        # 6. 좌우 안정성 (옆으로 기울지 않기)
        roll_angle = np.arctan2(up_vector[1], up_vector[2])
        lateral_reward = np.exp(-10 * abs(roll_angle))
        total_reward += self.weights['lateral_stability'] * lateral_reward
        reward_info['lateral_stability'] = lateral_reward

        # 7. 에너지 효율성
        motor_efforts = np.sum(np.square(data.ctrl))
        energy_penalty = motor_efforts
        total_reward += self.weights['energy'] * energy_penalty
        reward_info['energy'] = -energy_penalty

        # 8. 관절 한계 페널티
        joint_limit_penalty = self._compute_joint_limit_penalty(model, data)
        total_reward += self.weights['joint_limit'] * joint_limit_penalty
        reward_info['joint_limit'] = -joint_limit_penalty

        # 9. 좌우 대칭성 (2족 보행의 중요한 요소)
        symmetry_reward = self._compute_symmetry_reward(data)
        total_reward += self.weights['symmetry'] * symmetry_reward
        reward_info['symmetry'] = symmetry_reward

        # 10. 부드러운 동작
        if hasattr(self, '_last_action'):
            action_diff = np.sum(np.square(data.ctrl - self._last_action))
            smooth_reward = np.exp(-0.1 * action_diff)
            total_reward += self.weights['smooth_motion'] * smooth_reward
            reward_info['smooth_motion'] = smooth_reward

        self._last_action = data.ctrl.copy()

        return total_reward, reward_info

    def _quat_to_rotmat(self, quat):
        """Quaternion을 rotation matrix로 변환"""
        w, x, y, z = quat
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

    def _get_foot_contacts(self, model, data):
        """발 접촉 감지"""
        foot_names = ["FR", "FL", "RR", "RL"]
        contacts = []

        for foot_name in foot_names:
            try:
                foot_geom_id = model.geom(foot_name).id
                contact = False

                for i in range(data.ncon):
                    contact_geom1 = data.contact[i].geom1
                    contact_geom2 = data.contact[i].geom2

                    if contact_geom1 == foot_geom_id or contact_geom2 == foot_geom_id:
                        # 접촉력 확인
                        contact_force = np.linalg.norm(data.contact[i].force)
                        if contact_force > 0.1:  # 의미있는 접촉
                            contact = True
                            break

                contacts.append(1.0 if contact else 0.0)
            except:
                contacts.append(0.0)

        return contacts

    def _compute_joint_limit_penalty(self, model, data):
        """관절 한계 페널티 계산"""
        joint_pos = data.qpos[7:]  # 관절 위치
        joint_ranges = model.jnt_range[1:]  # 첫 번째는 root joint

        penalty = 0.0
        for i, pos in enumerate(joint_pos):
            if pos < joint_ranges[i, 0]:
                penalty += (joint_ranges[i, 0] - pos) ** 2
            elif pos > joint_ranges[i, 1]:
                penalty += (pos - joint_ranges[i, 1]) ** 2

        return penalty

    def _compute_symmetry_reward(self, data):
        """좌우 대칭성 보상"""
        # 관절 위치 (FR, FL, RR, RL 순서)
        joint_pos = data.qpos[7:19]

        # 왼쪽과 오른쪽 다리의 차이
        # FR(0-2) vs FL(3-5), RR(6-8) vs RL(9-11)
        front_diff = np.sum(np.abs(joint_pos[0:3] - joint_pos[3:6]))
        rear_diff = np.sum(np.abs(joint_pos[6:9] - joint_pos[9:12]))

        symmetry_error = front_diff + rear_diff
        symmetry_reward = np.exp(-2 * symmetry_error)

        return symmetry_reward


class EnhancedStableStandingReward:
    """
    개선된 2족 보행 보상 함수 - 동적 안정성과 물리적 제약 강화
    """
    def __init__(self):
        self.scale = 10.0
        self.target_height = 0.5
        self.target_vel = 0.3
        self.rear_feet = ["RR", "RL"]
        self.front_feet = ["FR", "FL"]

        # 학습 진행도 추적
        self.num_timesteps = 0
        self.success_buffer = deque(maxlen=100)
        self.joint_history = deque(maxlen=50)

        # 커리큘럼 학습 파라미터 (수정됨)
        self.curriculum_stage = 0
        self.height_tolerance = 0.25      # 기존 0.2 -> 0.25 (초기 허용 오차 완화)
        self.upright_threshold = 0.65   # 기존 0.7 -> 0.65 (초기 허용 오차 완화)

        # 이전 상태 저장
        self._prev_a = None
        self.prev_energy = None

    def compute_reward(self, model, data):
        """전체 보상 계산"""
        r = {}

        # 1) 자세 직립도 (개선된 버전)
        up_z = self._quat_to_rotmat(data.qpos[3:7])[:, 2][2]
        # 기울기에 대해 더 민감한 반응
        angle_from_vertical = np.arccos(np.clip(up_z, -1, 1))
        r["upright"] = np.exp(-5 * angle_from_vertical ** 2)

        # 2) 높이 (커리큘럼 적용)
        h = data.qpos[2]
        current_tolerance = self.height_tolerance * (0.5 + 0.5 * np.exp(-self.curriculum_stage))
        r["height"] = np.exp(-((h - self.target_height) / current_tolerance) ** 2)

        # 3) 발 접촉
        contacts = self._get_foot_contacts(model, data)
        if len(contacts) == 4:
            rear = (contacts[2] + contacts[3]) / 2.0
            front = (contacts[0] + contacts[1]) / 2.0
        else:
            rear = front = 0.0
        r["feet"] = rear - front

        # 4) 전진 속도
        v = data.qvel[0]
        r["forward_vel"] = np.exp(-((v - self.target_vel) / 0.15) ** 2)

        # 5) 좌우 속도 억제 (새로 추가)
        v_lat = np.linalg.norm(data.qvel[1:3])
        r["lateral_vel"] = np.exp(-(v_lat / 0.3) ** 2)

        # 6) 균형 (개선)
        ang_v = np.linalg.norm(data.qvel[3:6])
        r["stab_ang"] = np.exp(-(ang_v / 5.0) ** 2)

        # 7) COP 안정성 (새로 추가)
        r["cop_stab"] = self._cop_stability(model, data)

        # 8) ZMP 안정성 (새로 추가)
        r["zmp_stab"] = self._zmp_stability(model, data)

        # 9) 발 압력 분포 (새로 추가)
        r["pressure_dist"] = self._foot_pressure_distribution(model, data)

        # 10) 주기적 움직임 (새로 추가)
        r["gait_period"] = self._gait_periodicity_reward(data)

        # 11) 에너지 페널티 (적응형)
        effort = np.mean(np.square(data.ctrl))
        progress = min(1.0, self.num_timesteps / 1e6)
        energy_weight = 0.25 + 0.75 * progress
        r["energy"] = -energy_weight * math.tanh(effort / 50.0)

        # 12) 관절 한계 (개선)
        joint_penalty = self._joint_limit_violation_smooth(model, data)
        r["joint"] = -math.tanh(joint_penalty / 5.0)

        # 13) 부드러운 동작
        if self._prev_a is not None:
            diff = np.mean(np.square(data.ctrl - self._prev_a))
            r["smooth"] = math.exp(-(diff / 0.2) ** 2)
        else:
            r["smooth"] = 1.0
        self._prev_a = data.ctrl.copy()

        # 14) 대칭성
        r["symmetry"] = self._compute_symmetry_reward(data)

        # 15) 예측 안정성 (새로 추가)
        r["predictive"] = self._predictive_stability(model, data)

        # 16) 물리적 제약 (새로 추가)
        phys_reward, _ = self._physical_constraints_reward(model, data)
        r["physics"] = phys_reward

        # 총합 계산
        total = self.scale * sum(r.values()) / len(r)

        # 학습 진행도 업데이트
        self.num_timesteps += 1
        self._update_curriculum(r)

        return total, r

    def _update_curriculum(self, rewards):
        """커리큘럼 단계 업데이트 (수정됨)"""
        # 성공 판단 (높이 유지 & 직립 & 발 접촉)
        # 초기 성공 조건을 약간 완화하고, 단계별로 엄격하게 조정
        height_success_threshold = 0.65 + 0.05 * self.curriculum_stage
        upright_success_threshold = 0.75 + 0.05 * self.curriculum_stage

        success = (rewards.get("height", 0) > height_success_threshold and
                   rewards.get("upright", 0) > upright_success_threshold and
                   rewards.get("feet", 0) > 0.5)

        self.success_buffer.append(success)

        # 성공률이 80% 이상이면 다음 단계로
        if len(self.success_buffer) >= 50:
            success_rate = sum(self.success_buffer) / len(self.success_buffer)
            if success_rate > 0.8 and self.curriculum_stage < 5:
                self.curriculum_stage += 1
                # 감소폭 완화
                self.height_tolerance = max(0.1, self.height_tolerance * 0.85)
                self.upright_threshold = min(0.9, self.upright_threshold + 0.05)
                print(f"🎓 커리큘럼 진행: Stage {self.curriculum_stage}")

    def _cop_stability(self, model, data):
        """Center of Pressure 안정성"""
        try:
            # 간단한 근사: 뒷발 접촉점들의 중심
            rear_contacts = []
            for foot_name in self.rear_feet:
                foot_id = model.site(foot_name).id
                foot_pos = data.site_xpos[foot_id][:2]  # x, y 좌표
                rear_contacts.append(foot_pos)

            if len(rear_contacts) >= 2:
                cop = np.mean(rear_contacts, axis=0)
                # 지지 다각형 내부 판정 (간단화)
                center = np.mean(rear_contacts, axis=0)
                dist = np.linalg.norm(cop - center)
                max_dist = np.linalg.norm(rear_contacts[0] - rear_contacts[1]) / 2
                return np.exp(-5 * (dist / max_dist) ** 2)
            return 0.5
        except:
            return 0.5

    def _zmp_stability(self, model, data):
        """Zero Moment Point 안정성"""
        try:
            # 간단한 ZMP 근사
            com_pos = data.qpos[:3]
            com_vel = data.qvel[:3]
            gravity = model.opt.gravity[2]

            # ZMP x = COM_x - (COM_z / g) * COM_x_accel
            zmp_x = com_pos[0] - (com_pos[2] / abs(gravity)) * com_vel[0]
            zmp_y = com_pos[1] - (com_pos[2] / abs(gravity)) * com_vel[1]

            # 지지 다각형과의 거리
            support_center = np.array([com_pos[0], com_pos[1]])
            zmp_pos = np.array([zmp_x, zmp_y])
            dist = np.linalg.norm(zmp_pos - support_center)

            return np.exp(-3 * dist ** 2)
        except:
            return 0.5

    def _foot_pressure_distribution(self, model, data):
        """발 압력 분포의 균일성"""
        try:
            pressures = []
            for i in range(data.ncon):
                for foot_name in self.rear_feet:
                    foot_id = model.geom(foot_name).id
                    if data.contact[i].geom1 == foot_id or data.contact[i].geom2 == foot_id:
                        force = np.linalg.norm(data.contact[i].force)
                        pressures.append(force)

            if len(pressures) >= 2:
                # 압력 분포의 균일성
                var = np.var(pressures)
                return np.exp(-10 * var / (np.mean(pressures) ** 2 + 1e-6))
            return 0.5
        except:
            return 0.5

    def _gait_periodicity_reward(self, data, window=50):
        """보행의 주기성"""
        # 관절 각도 기록
        joint_angles = data.qpos[7:19]
        self.joint_history.append(joint_angles.copy())

        if len(self.joint_history) < window:
            return 0.5

        try:
            # 간단한 주기성 측정: 자기상관
            joint_data = np.array(list(self.joint_history)[-window:])
            mean_data = np.mean(joint_data, axis=0)
            centered = joint_data - mean_data

            # 자기상관 계산
            autocorr = np.correlate(centered[:, 6], centered[:, 6], mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            # 주기성 강도
            if len(autocorr) > 10:
                periodicity = np.max(autocorr[5:20]) / (autocorr[0] + 1e-6)
                return np.clip(periodicity, 0, 1)
            return 0.5
        except:
            return 0.5

    def _predictive_stability(self, model, data, horizon=5):
        """미래 안정성 예측"""
        try:
            # 현재 속도로 horizon 스텝 후 위치 예측
            com_pos = data.qpos[:3].copy()
            com_vel = data.qvel[:3].copy()
            dt = model.opt.timestep

            future_pos = com_pos + com_vel * dt * horizon

            # 예측 위치가 안정 범위 내에 있는지
            height_ok = 0.3 < future_pos[2] < 0.7
            lateral_ok = np.linalg.norm(future_pos[:2] - com_pos[:2]) < 0.2

            return float(height_ok and lateral_ok)
        except:
            return 0.5

    def _physical_constraints_reward(self, model, data):
        """물리적 제약 준수"""
        rewards = {}

        try:
            # 1. 각운동량 보존
            angular_vel = data.qvel[3:6]
            ang_momentum = np.linalg.norm(angular_vel)
            rewards['ang_momentum'] = np.exp(-0.1 * ang_momentum)

            # 2. 에너지 보존
            # 위치 에너지 + 운동 에너지
            height = data.qpos[2]
            vel = np.linalg.norm(data.qvel[:3])
            total_energy = 9.81 * height + 0.5 * vel ** 2

            if self.prev_energy is not None:
                energy_change = abs(total_energy - self.prev_energy)
                rewards['energy_conservation'] = np.exp(-10 * energy_change)
            else:
                rewards['energy_conservation'] = 1.0
            self.prev_energy = total_energy

        except:
            rewards = {'ang_momentum': 0.5, 'energy_conservation': 0.5}

        return np.mean(list(rewards.values())), rewards

    def _joint_limit_violation_smooth(self, model, data):
        """부드러운 관절 한계 페널티"""
        joint_pos = data.qpos[7:]
        joint_ranges = model.jnt_range[1:]

        penalty = 0.0
        margin = 0.1  # 여유 마진

        for i, pos in enumerate(joint_pos):
            lower, upper = joint_ranges[i]

            # 부드러운 페널티 함수
            if pos < lower + margin:
                violation = (lower + margin - pos) / margin
                penalty += 1 / (1 + np.exp(-10 * (violation - 0.5)))
            elif pos > upper - margin:
                violation = (pos - (upper - margin)) / margin
                penalty += 1 / (1 + np.exp(-10 * (violation - 0.5)))

        return penalty

    def _compute_symmetry_reward(self, data):
        """좌우 대칭성 보상"""
        joint_pos = data.qpos[7:19]

        # 왼쪽과 오른쪽 다리의 차이
        front_diff = np.sum(np.abs(joint_pos[0:3] - joint_pos[3:6]))
        rear_diff = np.sum(np.abs(joint_pos[6:9] - joint_pos[9:12]))

        symmetry_error = front_diff + rear_diff
        return np.exp(-2 * symmetry_error)

    # Helper methods
    def _quat_to_rotmat(self, quat):
        """Quaternion을 rotation matrix로 변환"""
        w, x, y, z = quat
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])

    def _get_foot_contacts(self, model, data):
        """발 접촉 감지"""
        foot_names = ["FR", "FL", "RR", "RL"]
        contacts = []

        for foot_name in foot_names:
            try:
                foot_geom_id = model.geom(foot_name).id
                contact = False

                for i in range(data.ncon):
                    if (data.contact[i].geom1 == foot_geom_id or
                            data.contact[i].geom2 == foot_geom_id):
                        contact_force = np.linalg.norm(data.contact[i].force)
                        if contact_force > 0.1:
                            contact = True
                            break

                contacts.append(1.0 if contact else 0.0)
            except:
                contacts.append(0.0)

        return contacts


class Go1StandingEnv(Go1MujocoEnv):
    """2족 보행을 학습하는 Go1 환경"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.standing_reward = EnhancedStableStandingReward()  # 개선된 보상 함수 사용
        self.episode_length = 0
        self.max_episode_length = 1000

        # --- 수정됨: 조기 종료 방지를 위한 설정 ---
        self.grace_steps = 30       # 초기 10 -> 30 스텝으로 유예 시간 증가
        self._post_reset = True     # reset 직후 상태 플래그
        self.consecutive_failures = 0
        self.failure_threshold = 5  # 5 프레임 연속으로 불안정할 경우에만 종료
        # --- 수정 끝 ---

        # 2족 보행을 위한 설정 변경
        self._healthy_z_range = (0.35, 0.65)  # 더 높은 자세
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))  # 약간 관대한 pitch
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))   # 더 엄격한 roll

        # Domain randomization 설정
        self.randomize_physics = True
        self.original_gravity = None

    def reset(self, seed=None, options=None):
        """환경 리셋 - 2족 보행 시작 자세로"""
        obs, info = super().reset(seed=seed, options=options)

        # original_gravity 초기화
        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # 2족 보행 시작 자세 설정
        self._set_standing_initial_pose()

        # Domain randomization
        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0
        self._post_reset = True

        # --- 추가됨: 연속 실패 카운터 리셋 ---
        self.consecutive_failures = 0
        # --- 추가 끝 ---

        # 보상 함수 리셋
        self.standing_reward.num_timesteps = getattr(self, 'total_timesteps', 0)

        return self._get_obs(), info

    def _apply_domain_randomization(self):
        """물리 파라미터 랜덤화"""
        if np.random.random() < 0.8:  # 80% 확률로 랜덤화
            # 중력 랜덤화
            gravity_scale = np.random.uniform(0.9, 1.1)
            self.model.opt.gravity[:] = self.original_gravity * gravity_scale

            # 마찰 계수 랜덤화
            friction_scale = np.random.uniform(0.8, 1.2)
            for i in range(self.model.ngeom):
                if hasattr(self.model, 'geom_friction'):
                    self.model.geom_friction[i, :] *= friction_scale

            # 질량 랜덤화 (작은 범위)
            mass_scale = np.random.uniform(0.95, 1.05)
            for i in range(self.model.nbody):
                if self.model.body_mass[i] > 0:
                    self.model.body_mass[i] *= mass_scale

    def _set_standing_initial_pose(self):
        """개선된 초기 자세 설정"""
        # 트렁크 위치
        self.data.qpos[0] = np.random.uniform(-0.1, 0.1)  # 약간의 랜덤성
        self.data.qpos[1] = np.random.uniform(-0.1, 0.1)
        self.data.qpos[2] = 0.5

        # 트렁크 자세 (약간의 노이즈 추가)
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = np.random.normal(0, 0.01, 3)

        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        self.data.qpos[3:7] /= quat_norm

        # 관절 초기화 (약간의 랜덤성)
        for i in range(self.model.nu):
            self.data.ctrl[i] = np.random.uniform(-0.01, 0.01)

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        # 뒷다리는 구부리고, 앞다리는 들어올린 자세
        joint_targets = np.zeros(12)
        # 뒷다리 (RR, RL) - 구부린 자세
        joint_targets[6:9] = [0.0, 0.8, -1.6]  # RR
        joint_targets[9:12] = [0.0, 0.8, -1.6]  # RL
        # 앞다리 (FR, FL) - 들어올린 자세
        joint_targets[0:3] = [0.0, 1.2, -2.0]  # FR
        joint_targets[3:6] = [0.0, 1.2, -2.0]  # FL

        # 노이즈 추가
        joint_targets += np.random.normal(0, 0.05, 12)
        self.data.qpos[7:19] = joint_targets

        # 시뮬레이터에 현재 상태 적용
        mujoco.mj_forward(self.model, self.data)

        # 발 높이 기반 z 보정
        foot_names = ["FR", "FL", "RR", "RL"]
        foot_ids = [self.model.site(name).id for name in foot_names]
        foot_zs = [self.data.site_xpos[site_id][2] for site_id in foot_ids]
        min_foot_z = min(foot_zs[2:])  # 뒷발만 고려

        target_clearance = 0.02
        self.data.qpos[2] -= (min_foot_z - target_clearance)

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        """환경 스텝 실행"""
        # 액션 실행
        self.do_simulation(action, self.frame_skip)

        # 관찰값 계산
        obs = self._get_obs()

        # 보상 계산
        reward, reward_info = self.standing_reward.compute_reward(self.model, self.data)

        # 종료 조건 확인
        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length

        self.episode_length += 1

        # post_reset 플래그 해제
        if self._post_reset and self.episode_length > 1:
            self._post_reset = False

        # 전체 타임스텝 추적
        if hasattr(self, 'total_timesteps'):
            self.total_timesteps += 1
        else:
            self.total_timesteps = 1

        info = {
            'episode_length': self.episode_length,
            'standing_reward': reward,
            'standing_success': self._is_standing_successful(),
            'curriculum_stage': self.standing_reward.curriculum_stage,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _is_terminated(self):
        """개선된 종료 조건 (연속 실패 기반, 수정됨)"""
        # 리셋 직후 또는 그레이스 기간 중에는 종료 조건 무시하고 카운터 초기화
        if self.episode_length < self.grace_steps or self._post_reset:
            self.consecutive_failures = 0
            return False

        is_failure = False

        # 기본 건강 상태 확인
        if not self.is_healthy:
            is_failure = True

        # z축이 지나치게 기울어진 경우
        trunk_quat = self.data.qpos[3:7]
        up_vector = self.standing_reward._quat_to_rotmat(trunk_quat)[:, 2]
        min_upright = 0.7 + 0.1 * min(self.standing_reward.curriculum_stage / 5, 1.0)
        if up_vector[2] < min_upright:
            is_failure = True

        # 높이가 너무 낮은 경우
        if self.data.qpos[2] < 0.3:
            is_failure = True

        # 너무 빠른 회전
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        if angular_vel > 10.0:
            is_failure = True

        # 실패 상태에 따라 카운터 업데이트
        if is_failure:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        # 연속 실패 횟수가 임계값을 넘으면 종료
        if self.consecutive_failures >= self.failure_threshold:
            return True

        return False

    def _is_standing_successful(self):
        """2족 보행 성공 판정 (개선)"""
        trunk_height = self.data.qpos[2]
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.standing_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]

        # 발 접촉 확인
        foot_contacts = self.standing_reward._get_foot_contacts(self.model, self.data)

        # 커리큘럼에 따른 성공 기준
        stage = self.standing_reward.curriculum_stage

        # 기본 조건
        height_ok = 0.4 < trunk_height < 0.6
        upright_ok = up_vector[2] > (0.85 + 0.02 * stage)
        rear_feet_contact = foot_contacts[2] > 0.5 and foot_contacts[3] > 0.5
        front_feet_up = foot_contacts[0] < 0.3 and foot_contacts[1] < 0.3

        # 안정성 조건
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        stable = angular_vel < 2.0

        # 지속 시간 조건 (스테이지별로 다름)
        min_duration = 50 + 20 * stage
        duration_ok = self.episode_length > min_duration

        # 전진 조건 (높은 스테이지에서만)
        forward_vel = self.data.qvel[0]
        if stage >= 3:
            moving_forward = forward_vel > 0.1
        else:
            moving_forward = True  # 초기에는 전진 조건 무시

        return (height_ok and upright_ok and rear_feet_contact and
                front_feet_up and duration_ok and stable and moving_forward)


class GradualStandingEnv(Go1StandingEnv):
    """점진적으로 4족에서 2족 보행으로 전환하는 환경 (수정됨)"""

    def __init__(self, curriculum_stage=0, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_curriculum()

    def _setup_curriculum(self):
        """커리큘럼 단계별 설정"""
        if self.curriculum_stage == 0:
            # Stage 0: 안정적인 4족 보행에 집중
            self.reward_weights = {
                'quadruped': 0.9,  # 4족 보행 보상 비중을 높임
                'standing': 0.1
            }
            self.grace_steps = 30  # 유예 시간 증가
            self.max_episode_length = 500
            # Stage 0에서는 4족 보행에 맞는 건강 범위를 사용
            self._healthy_z_range = (0.2, 0.5)
            self._healthy_pitch_range = (-np.deg2rad(30), np.deg2rad(30))
            self._healthy_roll_range = (-np.deg2rad(25), np.deg2rad(25))

        elif self.curriculum_stage == 1:
            # Stage 1: 앞발 들기 연습
            self.reward_weights = {
                'quadruped': 0.6,  # 점진적으로 비중 변경
                'standing': 0.4
            }
            self.grace_steps = 20
            self.max_episode_length = 750
            # 2족 보행 자세로 전환
            self._healthy_z_range = (0.3, 0.6)
            self._healthy_pitch_range = (-np.deg2rad(20), np.deg2rad(20))
            self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

        elif self.curriculum_stage == 2:
            # Stage 2: 2족 자세 유지
            self.reward_weights = {
                'quadruped': 0.2,
                'standing': 0.8
            }
            self.grace_steps = 10
            self.max_episode_length = 1000
            # 2족 보행 자세에 더 가까워짐
            self._healthy_z_range = (0.35, 0.65)
            self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
            self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        else:
            # Stage 3+: 완전한 2족 보행
            self.reward_weights = {
                'quadruped': 0.0,
                'standing': 1.0
            }
            self.grace_steps = 5
            self.max_episode_length = 1500
            # 최종 2족 보행 자세
            self._healthy_z_range = (0.35, 0.65)
            self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
            self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

    def step(self, action):
        """혼합 보상을 사용한 스텝"""
        # 기본 스텝 실행
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        # 2족 보행 보상
        standing_reward, standing_info = self.standing_reward.compute_reward(self.model, self.data)

        # 4족 보행 보상 (필요시)
        if self.reward_weights['quadruped'] > 0:
            quadruped_reward = self._compute_quadruped_reward()
        else:
            quadruped_reward = 0

        # 혼합 보상
        total_reward = (self.reward_weights['standing'] * standing_reward +
                        self.reward_weights['quadruped'] * quadruped_reward)

        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length

        self.episode_length += 1

        info = {
            'episode_length': self.episode_length,
            'standing_reward': standing_reward,
            'quadruped_reward': quadruped_reward,
            'total_reward': total_reward,
            'curriculum_stage': self.curriculum_stage,
            'standing_success': self._is_standing_successful(),
            **standing_info
        }

        return obs, total_reward, terminated, truncated, info

    def _compute_quadruped_reward(self):
        """간단한 4족 보행 보상"""
        # 전진 속도
        forward_vel = self.data.qvel[0]
        vel_reward = np.clip(forward_vel, 0, 1)

        # 안정성
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        stability_reward = np.exp(-angular_vel)

        # 에너지 효율
        energy = np.sum(np.square(self.data.ctrl))
        energy_reward = np.exp(-0.01 * energy)

        return vel_reward + 0.5 * stability_reward + 0.1 * energy_reward

    def advance_curriculum(self, success_rate):
        """성공률에 따라 커리큘럼 진행"""
        if success_rate > 0.8 and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            self._setup_curriculum()
            # 보상 함수의 커리큘럼도 업데이트
            self.standing_reward.curriculum_stage = self.curriculum_stage
            print(f"🎓 커리큘럼 진행: Stage {self.curriculum_stage}")
            return True
        return False