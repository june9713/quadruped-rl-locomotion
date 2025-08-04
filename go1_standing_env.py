#!/usr/bin/env python3
"""
Go1 4족 정상 서있기 환경 - 자연스러운 4족 자세에서 시작 (관찰 공간 호환성 개선)
"""

import numpy as np
import mujoco
from go1_mujoco_env import Go1MujocoEnv
import math
from collections import deque
from gymnasium import spaces
import os
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO


# visual_train.py에서 import할 수 있도록 환경 이름 추가
__all__ = ['Go1StandingEnv', 'GradualStandingEnv', 'StandingReward', 
           'BipedWalkingReward', 'BipedalWalkingEnv', 'BipedalCurriculumEnv',
           'create_compatible_env']


class StandingReward:
    """4족 정상 서있기를 위한 보상 함수"""

    def __init__(self):
        # 보상 가중치들 - 4족 서있기 최적화
        self.weights = {
            'upright': 12.0,        # 똑바로 서있기
            'height': 8.0,          # 적절한 높이 유지 (4족 기준)
            'balance': 10.0,        # 균형 유지
            'foot_contact': 8.0,    # 모든 발이 지면에 접촉
            'forward_vel': 0.0,     # 전진 속도 - 제거 (제자리 서기)
            'lateral_stability': 6.0, # 좌우 안정성
            'energy': -0.03,        # 에너지 효율
            'joint_limit': -3.0,    # 관절 한계 페널티
            'symmetry': 4.0,        # 좌우 대칭성
            'smooth_motion': 3.0    # 부드러운 동작
        }

    def compute_reward(self, model, data):
        """4족 서있기 보상 계산"""
        total_reward = 0.0
        reward_info = {}
        
        # 1. 높이 보상 (4족 서있기 목표 높이)
        trunk_height = data.qpos[2]
        target_height = 0.31  # 4족 서있기 목표 높이
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-10 * height_error) if 0.22 < trunk_height < 0.40 else 0
        total_reward += self.weights['height'] * height_reward
        reward_info['height'] = height_reward
        
        # 2. 직립도 보상
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        upright_reward = np.exp(-5 * (1 - up_vector[2])**2) if up_vector[2] > 0.85 else 0
        total_reward += self.weights['upright'] * upright_reward
        reward_info['upright'] = upright_reward
        
        # 3. 발 접촉 보상
        foot_contacts = self._get_foot_contacts(model, data)
        contact_reward = np.mean(foot_contacts)  # 모든 발이 접촉할수록 높은 보상
        total_reward += self.weights['foot_contact'] * contact_reward
        reward_info['foot_contact'] = contact_reward
        
        # 4. 안정성 보상 (속도 제한)
        linear_vel = np.linalg.norm(data.qvel[:3])
        angular_vel = np.linalg.norm(data.qvel[3:6])
        stability_reward = np.exp(-2 * (linear_vel + angular_vel))
        total_reward += self.weights['balance'] * stability_reward
        reward_info['balance'] = stability_reward
        
        # 5. 좌우 안정성 보상
        roll_angle = np.arctan2(trunk_rotation_matrix[2, 1], trunk_rotation_matrix[2, 2])
        lateral_reward = np.exp(-5 * roll_angle**2)
        total_reward += self.weights['lateral_stability'] * lateral_reward
        reward_info['lateral_stability'] = lateral_reward
        
        # 6. 에너지 효율 페널티
        joint_vel = data.qvel[7:]  # 관절 속도
        energy_penalty = np.sum(joint_vel**2)
        total_reward += self.weights['energy'] * energy_penalty
        reward_info['energy'] = -energy_penalty
        
        # 7. 관절 한계 페널티
        joint_pos = data.qpos[7:]
        joint_ranges = model.jnt_range[1:]  # 첫 번째는 root joint
        limit_penalty = 0.0
        for i, pos in enumerate(joint_pos):
            if pos < joint_ranges[i, 0]:
                limit_penalty += (joint_ranges[i, 0] - pos) ** 2
            elif pos > joint_ranges[i, 1]:
                limit_penalty += (pos - joint_ranges[i, 1]) ** 2
        total_reward += self.weights['joint_limit'] * limit_penalty
        reward_info['joint_limit'] = -limit_penalty
        
        # 8. 좌우 대칭성 보상
        joint_pos = data.qpos[7:19]
        front_diff = np.sum(np.abs(joint_pos[0:3] - joint_pos[3:6]))
        rear_diff = np.sum(np.abs(joint_pos[6:9] - joint_pos[9:12]))
        symmetry_error = front_diff + rear_diff
        symmetry_reward = np.exp(-2 * symmetry_error)
        total_reward += self.weights['symmetry'] * symmetry_reward
        reward_info['symmetry'] = symmetry_reward
        
        # 9. 부드러운 동작 보상
        if hasattr(self, '_prev_joint_pos'):
            joint_acc = np.sum((joint_pos - self._prev_joint_pos)**2)
            smooth_reward = np.exp(-5 * joint_acc)
            total_reward += self.weights['smooth_motion'] * smooth_reward
            reward_info['smooth_motion'] = smooth_reward
        else:
            reward_info['smooth_motion'] = 0.0
        
        self._prev_joint_pos = joint_pos.copy()
        
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


class BipedWalkingReward:
    """2족 보행을 위한 보상 함수"""
    
    def __init__(self):
        self.weights = {
            # 2족 보행 핵심 보상
            'bipedal_posture': 15.0,      # 2족 자세 유지
            'height': 10.0,                # 적절한 높이 (높게)
            'front_feet_up': 12.0,         # 앞발 들기
            'rear_feet_contact': 8.0,      # 뒷발만 접촉
            
            # 균형 관련
            'com_over_support': 10.0,      # 무게중심이 뒷발 위에
            'lateral_stability': 6.0,      # 좌우 안정성
            'angular_stability': 5.0,      # 각속도 안정성
            
            # 동작 관련 - 수직 자세 강조
            'torso_upright': 12.0,         # 상체 직립 (가중치 증가)
            'smooth_motion': 3.0,          # 부드러운 동작
            'forward_lean': 0.0,           # 전방 기울기 제거 (수직 목표)
            
            # 페널티
            'energy': -0.02,               # 에너지 효율
            'joint_limit': -2.0,           # 관절 한계
            'excessive_motion': -3.0       # 과도한 움직임
        }
        
        # 2족 보행 단계별 목표
        self.bipedal_stages = {
            'prepare': 0,      # 준비 자세
            'lifting': 1,      # 앞발 들기
            'balancing': 2,    # 균형 잡기
            'standing': 3      # 2족 서기
        }
        self.current_stage = 'prepare'

    def _get_foot_contacts(self, model, data):
            """발 접촉 감지 - StandingReward와 동일한 메서드"""
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


    def compute_reward(self, model, data):
        """2족 보행 보상 계산"""
        total_reward = 0.0
        reward_info = {}
        
        # 1. 2족 자세 보상
        front_feet_height = self._get_front_feet_height(model, data)
        rear_feet_contact = self._get_rear_feet_contact(model, data)
        
        # 앞발이 높이 들려있고, 뒷발만 접촉
        bipedal_score = np.mean(front_feet_height) * np.mean(rear_feet_contact)
        total_reward += self.weights['bipedal_posture'] * bipedal_score
        reward_info['bipedal_posture'] = bipedal_score
        
        # 2. 높이 보상 (2족은 더 높아야 함)
        trunk_height = data.qpos[2]
        target_height = 0.45  # 2족 목표 높이
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-10 * height_error) if trunk_height > 0.3 else 0
        total_reward += self.weights['height'] * height_reward
        reward_info['height'] = height_reward
        
        # 3. 앞발 들기 보상
        min_lift_height = 0.05  # 최소 5cm 이상
        front_feet_up_reward = 0
        for height in front_feet_height:
            if height > min_lift_height:
                front_feet_up_reward += np.tanh(height / 0.1)  # 부드러운 보상
        front_feet_up_reward /= 2  # 정규화
        total_reward += self.weights['front_feet_up'] * front_feet_up_reward
        reward_info['front_feet_up'] = front_feet_up_reward
        
        # 4. 무게중심이 뒷발 위에 있는지
        com_position = self._get_com_position(model, data)
        rear_feet_positions = self._get_rear_feet_positions(model, data)
        
        # 무게중심의 x,y가 뒷발 사이에 있는지 확인
        com_score = self._compute_com_over_support(com_position, rear_feet_positions)
        total_reward += self.weights['com_over_support'] * com_score
        reward_info['com_over_support'] = com_score
        
        # 5. 상체 직립 보상 - 수정됨: 완전히 수직을 목표로
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        # 수직 자세를 목표로 (0도, 허용 오차 ±5도)
        current_pitch = np.arcsin(-trunk_rotation_matrix[2, 0])
        
        # pitch가 0에 가까울수록 높은 보상
        pitch_penalty = abs(current_pitch)
        
        # up_vector[2]가 1에 가까울수록 (완전히 수직일수록) 높은 보상
        verticality_reward = up_vector[2] ** 3  # 3제곱으로 더 강한 수직 유도
        
        # 두 가지 요소 결합: 수직도 + pitch 각도
        torso_reward = verticality_reward * np.exp(-10 * pitch_penalty)
        
        # 추가 보너스: 거의 수직일 때 (±3도 이내)
        if abs(current_pitch) < np.deg2rad(3):
            torso_reward *= 1.5
        
        total_reward += self.weights['torso_upright'] * torso_reward
        reward_info['torso_upright'] = torso_reward
        
        # 6. 안정성 보상
        angular_vel = data.qvel[3:6]
        angular_stability = np.exp(-2 * np.linalg.norm(angular_vel))
        total_reward += self.weights['angular_stability'] * angular_stability
        reward_info['angular_stability'] = angular_stability
        
        # 7. 에너지 페널티 (2족은 더 많은 토크 허용)
        motor_efforts = np.sum(np.square(data.ctrl))
        energy_penalty = motor_efforts * 0.5  # 페널티 완화
        total_reward += self.weights['energy'] * energy_penalty
        reward_info['energy'] = -energy_penalty
        
        # 8. 단계별 보너스 - 수정됨: 수직 자세 추가 고려
        stage_bonus = self._compute_stage_bonus(front_feet_height, rear_feet_contact, 
                                               trunk_height, current_pitch)
        total_reward += stage_bonus
        reward_info['stage_bonus'] = stage_bonus
        
        return total_reward, reward_info

    def _get_front_feet_height(self, model, data):
        """앞발 높이 계산"""
        front_feet_heights = []
        for foot_name in ["FR", "FL"]:
            try:
                foot_site_id = model.site(foot_name).id
                foot_pos = data.site_xpos[foot_site_id]
                front_feet_heights.append(foot_pos[2])  # z 좌표
            except:
                front_feet_heights.append(0.0)
        return front_feet_heights

    def _get_rear_feet_contact(self, model, data):
        """뒷발 접촉 상태"""
        rear_contacts = []
        for foot_name in ["RR", "RL"]:
            try:
                foot_geom_id = model.geom(foot_name).id
                contact = False
                for i in range(data.ncon):
                    contact_geom1 = data.contact[i].geom1
                    contact_geom2 = data.contact[i].geom2
                    if contact_geom1 == foot_geom_id or contact_geom2 == foot_geom_id:
                        contact_force = np.linalg.norm(data.contact[i].force)
                        if contact_force > 0.1:
                            contact = True
                            break
                rear_contacts.append(1.0 if contact else 0.0)
            except:
                rear_contacts.append(0.0)
        return rear_contacts

    def _get_com_position(self, model, data):
        """무게중심 위치"""
        return data.xpos[1]  # root body의 위치

    def _get_rear_feet_positions(self, model, data):
        """뒷발 위치들"""
        rear_positions = []
        for foot_name in ["RR", "RL"]:
            try:
                foot_site_id = model.site(foot_name).id
                foot_pos = data.site_xpos[foot_site_id]
                rear_positions.append(foot_pos[:2])  # x, y 좌표만
            except:
                rear_positions.append([0.0, 0.0])
        return rear_positions

    def _compute_com_over_support(self, com_position, rear_feet_positions):
        """무게중심이 뒷발 위에 있는지 계산"""
        if len(rear_feet_positions) < 2:
            return 0.0
        
        # 뒷발 사이의 중심점
        rear_center = np.mean(rear_feet_positions, axis=0)
        
        # 무게중심과 뒷발 중심의 거리
        com_xy = com_position[:2]
        distance = np.linalg.norm(com_xy - rear_center)
        
        # 거리가 가까울수록 높은 보상
        return np.exp(-5 * distance)

    def _compute_stage_bonus(self, front_feet_height, rear_feet_contact, trunk_height, current_pitch):
        """단계별 보너스 계산 - 수직 자세 추가 고려"""
        stage_bonus = 0.0
        
        # 준비 단계: 높이 유지
        if trunk_height > 0.35:
            stage_bonus += 2.0
        
        # 들기 단계: 앞발 들기
        if np.mean(front_feet_height) > 0.03:
            stage_bonus += 3.0
        
        # 균형 단계: 뒷발 접촉 유지
        if np.mean(rear_feet_contact) > 0.8:
            stage_bonus += 2.0
        
        # 수직 자세 보너스 (새로 추가)
        if abs(current_pitch) < np.deg2rad(10):  # 10도 이내
            stage_bonus += 3.0
        if abs(current_pitch) < np.deg2rad(5):   # 5도 이내
            stage_bonus += 5.0
        
        # 2족 서기 단계: 모든 조건 만족 (수직 조건 추가)
        if (trunk_height > 0.4 and 
            np.mean(front_feet_height) > 0.05 and 
            np.mean(rear_feet_contact) > 0.9 and
            abs(current_pitch) < np.deg2rad(5)):  # 5도 이내 수직
            stage_bonus += 10.0  # 보너스 증가
        
        return stage_bonus

    def _quat_to_rotmat(self, quat):
        """Quaternion을 rotation matrix로 변환"""
        w, x, y, z = quat
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])


class Go1StandingEnv(Go1MujocoEnv):
    """4족 정상 서있기 환경 - 자연스러운 4족 자세에서 시작 (관찰 공간 호환성 개선)"""

    def __init__(self, **kwargs):
        # 허용되지 않는 파라미터들 제거
        filtered_kwargs = {}
        allowed_params = {
            'randomize_physics', 'render_mode', 'frame_skip', 
            'observation_space', 'default_camera_config'
        }
        
        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value
        
        # ✅ 관찰 공간 호환성을 위한 설정
        self._use_base_observation = kwargs.get('use_base_observation', False)
        
        # 부모 클래스 초기화 (필터링된 kwargs 사용)
        super().__init__(**filtered_kwargs)
        
        self.standing_reward = StandingReward()
        self.episode_length = 0
        self.max_episode_length = 1000

        # 4족 서있기를 위한 건강 상태 조건
        self._healthy_z_range = (0.22, 0.40)  # 4족 서있기 높이 범위
        self._healthy_pitch_range = (-np.deg2rad(20), np.deg2rad(20))
        self._healthy_roll_range = (-np.deg2rad(20), np.deg2rad(20))

        # Domain randomization 설정
        self.randomize_physics = kwargs.get('randomize_physics', True)
        self.original_gravity = None

        # ✅ 점진적 노이즈 감소를 위한 훈련 진행도 추적
        self.total_timesteps = 0
        self.max_training_timesteps = 5_000_000  # 예상 총 훈련 스텝
        
        # ✅ 관찰 공간 재설정
        if self._use_base_observation:
            # 기본 Go1MujocoEnv와 동일한 관찰 공간 사용 (45차원)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=self._get_base_obs().shape, 
                dtype=np.float64
            )
            print(f"🔄 호환 모드: 기본 관찰 공간({self._get_base_obs().shape[0]}차원) 사용")
        else:
            # 확장된 관찰 공간 사용 (56차원)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=self._get_extended_obs().shape, 
                dtype=np.float64
            )


    def _get_adaptive_noise_scale(self):
        """훈련 진행도에 따라 노이즈 스케일을 점진적으로 감소"""
        # 진행도 계산 (0.0 ~ 1.0)
        progress = min(self.total_timesteps / self.max_training_timesteps, 1.0)
        
        # 초기에는 매우 큰 노이즈, 점차 감소
        # 이전보다 더 큰 초기 노이즈와 더 급격한 감소
        initial_scale = 0.35  # 초기 노이즈 증가 (이전 0.25 -> 0.35)
        final_scale = 0.015   # 최종 노이즈 감소 (이전 0.02 -> 0.015)
        
        # 다단계 감소 함수 사용
        if progress < 0.2:
            # 초기 20%: 매우 큰 노이즈 유지
            noise_scale = initial_scale
        elif progress < 0.5:
            # 20-50%: 천천히 감소
            t = (progress - 0.2) / 0.3
            noise_scale = initial_scale * (1 - t * 0.3)  # 30% 감소
        elif progress < 0.8:
            # 50-80%: 더 빠르게 감소
            t = (progress - 0.5) / 0.3
            mid_scale = initial_scale * 0.7
            noise_scale = mid_scale * (1 - t * 0.6)  # 60% 추가 감소
        else:
            # 80-100%: 최종 미세 조정
            t = (progress - 0.8) / 0.2
            low_scale = initial_scale * 0.28
            noise_scale = low_scale * (1 - t * 0.9) + final_scale * t
        
        return noise_scale


    def _get_extended_obs(self):
        """확장된 관찰 상태 (2족 보행용 추가 정보 포함)"""
        # 기본 정보 (45차원)
        base_obs = self._get_base_obs()
        
        # 2족 보행 특화 정보 추가
        # 1. 발 높이 정보
        foot_heights = np.array([
            self._get_foot_height('FR'),
            self._get_foot_height('FL'),
            self._get_foot_height('RR'),
            self._get_foot_height('RL')
        ])
        
        # 2. 발 접촉 정보 - 환경별 보상 객체 사용
        reward_obj = self._get_reward_object()
        if reward_obj:
            foot_contacts = np.array(reward_obj._get_foot_contacts(self.model, self.data))
        else:
            # 보상 객체가 없으면 직접 계산
            foot_contacts = np.array(self._get_foot_contacts_direct())
        
        # 3. 상체 기울기 (pitch, roll)
        trunk_quat = self.data.qpos[3:7]
        pitch, roll = self._quat_to_euler(trunk_quat)[:2]
        
        # 4. 목표 자세 정보 (2족 서기 목표)
        target_height = 0.45  # 2족 목표 높이
        height_error = abs(self.data.qpos[2] - target_height)
        
        # 추가 정보 결합 (11차원)
        extended_info = np.concatenate([
            foot_heights,           # 4차원
            foot_contacts,          # 4차원  
            [pitch, roll],          # 2차원
            [height_error]          # 1차원
        ])
        
        # 전체 관찰 상태 = 기본(45) + 확장(11) = 56차원
        return np.concatenate([base_obs, extended_info])

    def _get_base_obs(self):
        """기본 Go1MujocoEnv와 호환되는 관찰 상태 (45차원)"""
        # 부모 클래스의 관찰 방법 사용
        return super()._get_obs()
    
    def _get_obs(self):
        """관찰 상태 반환 - 호환성 모드에 따라 선택"""
        if self._use_base_observation:
            return self._get_base_obs()
        else:
            return self._get_extended_obs()

    def _get_foot_height(self, foot_name):
        """발 높이 계산"""
        try:
            foot_site_id = self.model.site(foot_name).id
            foot_pos = self.data.site_xpos[foot_site_id]
            return foot_pos[2]  # z 좌표
        except:
            return 0.0

    def _quat_to_euler(self, quat):
        """Quaternion을 Euler angles로 변환"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def _set_bipedal_ready_pose(self):
        """2족 보행 준비 자세 설정 - 적응적 노이즈 적용"""
        
        # 적응적 노이즈 스케일 계산
        noise_scale = self._get_adaptive_noise_scale()
        
        # 1. 트렁크 위치 - 적응적 노이즈
        position_noise = noise_scale * 0.15
        self.data.qpos[0] = np.random.uniform(-position_noise, position_noise)  # x
        self.data.qpos[1] = np.random.uniform(-position_noise, position_noise)  # y
        
        # 높이 변동
        height_base = 0.35
        height_noise = noise_scale * 0.12
        self.data.qpos[2] = height_base + np.random.uniform(-height_noise, height_noise)

        # 2. 트렁크 자세 - 수직 자세를 목표로 다양한 시도
        angle_noise = noise_scale * 0.5  # 2족은 더 다양한 각도 필요
        
        # 기본 수직 자세(0도)에서 시작, 노이즈 추가
        base_pitch = 0.0  # 수직 자세 목표
        
        # 초기 훈련에서는 다양한 pitch 각도 시도
        if noise_scale > 0.15:
            # 때때로 극단적인 각도도 시도 (학습 다양성)
            if np.random.random() < 0.3:
                # 30% 확률로 더 큰 범위 시도
                pitch_angle = np.random.uniform(-angle_noise * 1.5, angle_noise * 1.5)
            else:
                pitch_angle = base_pitch + np.random.uniform(-angle_noise, angle_noise)
        else:
            # 후반 훈련에서는 수직 근처만 시도
            pitch_angle = base_pitch + np.random.uniform(-angle_noise, angle_noise)
        
        roll_angle = np.random.uniform(-angle_noise*0.5, angle_noise*0.5)
        yaw_angle = np.random.uniform(-angle_noise*0.3, angle_noise*0.3)
        
        # 오일러 각을 쿼터니언으로 변환
        r = Rotation.from_euler('xyz', [roll_angle, pitch_angle, yaw_angle])
        quat = r.as_quat()  # [x, y, z, w] 순서
        
        # MuJoCo는 [w, x, y, z] 순서 사용
        self.data.qpos[3] = quat[3]  # w
        self.data.qpos[4] = quat[0]  # x
        self.data.qpos[5] = quat[1]  # y
        self.data.qpos[6] = quat[2]  # z

        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        self.data.qpos[3:7] /= quat_norm

        # 3. 2족 보행 준비 관절 각도 - 매우 다양한 범위의 노이즈
        joint_noise_scale = noise_scale * 1.5  # 2족은 더 큰 노이즈 필요
        
        # 기본 2족 준비 자세 - 수직 자세를 고려한 관절 각도
        base_joint_targets = np.array([
            # 앞다리 (FR, FL) - 들기 준비
            0.0, 0.5, -1.0,    # FR
            0.0, 0.5, -1.0,    # FL
            
            # 뒷다리 (RR, RL) - 지지 준비
            0.0, 0.6, -1.2,    # RR
            0.0, 0.6, -1.2     # RL
        ])
        
        # 각 관절마다 매우 다른 범위의 노이즈 적용
        joint_noise = np.zeros(12)
        
        for i in range(12):
            # 관절 종류와 다리 위치에 따라 다른 노이즈 범위
            if i < 6:  # 앞다리
                if i % 3 == 0:  # Hip joints (0, 3)
                    range_multiplier = np.random.uniform(0.3, 1.2)
                elif i % 3 == 1:  # Knee joints (1, 4) - 앞다리 무릎은 큰 범위
                    range_multiplier = np.random.uniform(1.0, 3.0)
                else:  # Ankle joints (2, 5) - 앞다리 발목도 큰 범위
                    range_multiplier = np.random.uniform(0.8, 2.5)
            else:  # 뒷다리
                if i % 3 == 0:  # Hip joints (6, 9)
                    range_multiplier = np.random.uniform(0.4, 1.0)
                elif i % 3 == 1:  # Knee joints (7, 10)
                    range_multiplier = np.random.uniform(0.6, 1.8)
                else:  # Ankle joints (8, 11)
                    range_multiplier = np.random.uniform(0.5, 1.5)
            
            # 기본 노이즈
            joint_noise[i] = np.random.uniform(-joint_noise_scale * range_multiplier, 
                                             joint_noise_scale * range_multiplier)
        
        # 초기 훈련에서는 극단적인 자세도 시도
        if noise_scale > 0.15:
            # 앞다리를 아예 높이 들어올리는 시도
            if np.random.random() < 0.3:  # 30% 확률
                extreme_lift = np.random.uniform(2.0, 4.0)
                joint_noise[1] += np.random.uniform(-1.2, -0.3) * extreme_lift  # FR 무릎 더 굽히기
                joint_noise[2] += np.random.uniform(0.3, 1.2) * extreme_lift    # FR 발목 더 들기
                joint_noise[4] += np.random.uniform(-1.2, -0.3) * extreme_lift  # FL 무릎 더 굽히기
                joint_noise[5] += np.random.uniform(0.3, 1.2) * extreme_lift    # FL 발목 더 들기
            
            # 뒷다리 극단적 자세 - 수직 자세 유지를 위해
            if np.random.random() < 0.4:  # 40% 확률
                extreme_support = np.random.uniform(1.5, 3.0)
                joint_noise[7] += np.random.uniform(-0.6, 0.6) * extreme_support   # RR 무릎
                joint_noise[8] += np.random.uniform(-0.6, 0.6) * extreme_support   # RR 발목
                joint_noise[10] += np.random.uniform(-0.6, 0.6) * extreme_support  # RL 무릎
                joint_noise[11] += np.random.uniform(-0.6, 0.6) * extreme_support  # RL 발목
            
            # 비대칭 극단적 움직임
            if np.random.random() < 0.25:  # 25% 확률
                # 한쪽 앞다리만 극단적으로 들기
                if np.random.random() < 0.5:
                    joint_noise[0:3] += np.random.uniform(-0.5, 0.5) * 2.0  # FR 극단적
                else:
                    joint_noise[3:6] += np.random.uniform(-0.5, 0.5) * 2.0  # FL 극단적
        
        # 최종 관절 각도
        joint_targets = base_joint_targets + joint_noise
        
        # 관절 한계 내로 클리핑 (2족은 한계에 더 가깝게)
        joint_ranges = self.model.jnt_range[1:]
        for i in range(12):
            joint_targets[i] = np.clip(joint_targets[i], 
                                      joint_ranges[i, 0] * 0.95, 
                                      joint_ranges[i, 1] * 0.95)
        
        self.data.qpos[7:19] = joint_targets

        # 4. 속도 노이즈 - 2족은 더 다양한 속도
        vel_noise_scale = noise_scale * 0.4
        
        for i in range(len(self.data.qvel)):
            if i < 3:  # 선속도
                vel_range = np.random.uniform(0.5, 2.0)
            elif i < 6:  # 각속도 - 2족은 더 큰 각속도 변화
                vel_range = np.random.uniform(0.8, 3.0)
            else:  # 관절 속도
                vel_range = np.random.uniform(0.5, 2.5)
            
            self.data.qvel[i] = np.random.normal(0, vel_noise_scale * vel_range)
        
        self.data.qacc[:] = 0.0

        # 5. 제어 입력 초기화
        self.data.ctrl[:] = 0.0

        # 6. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)

        # 7. 발이 지면에 접촉하도록 높이 자동 조정
        self._auto_adjust_height_for_ground_contact()

    def _set_natural_standing_pose(self):
        """자연스러운 4족 서있기 자세 설정 - 적응적 노이즈 적용"""
        
        # 적응적 노이즈 스케일 계산
        noise_scale = self._get_adaptive_noise_scale()
        
        # 1. 트렁크 위치 설정 - 적응적 노이즈
        position_noise = noise_scale * 0.2  # 위치 노이즈 (초기 5cm -> 최종 0.4cm)
        self.data.qpos[0] = np.random.uniform(-position_noise, position_noise)  # x
        self.data.qpos[1] = np.random.uniform(-position_noise, position_noise)  # y
        
        # 높이도 약간의 변동 추가
        height_base = 0.30
        height_noise = noise_scale * 0.15  # 높이 노이즈 (초기 3.75cm -> 최종 0.3cm)
        self.data.qpos[2] = height_base + np.random.uniform(-height_noise, height_noise)

        # 2. 트렁크 자세 - 적응적 각도 노이즈
        angle_noise = noise_scale * 0.4  # 각도 노이즈 (초기 10도 -> 최종 0.8도)
        
        # 랜덤한 초기 자세 (pitch, roll, yaw에 노이즈)
        pitch_noise = np.random.uniform(-angle_noise, angle_noise)
        roll_noise = np.random.uniform(-angle_noise, angle_noise)
        yaw_noise = np.random.uniform(-angle_noise, angle_noise)
        
        # 오일러 각을 쿼터니언으로 변환
        r = Rotation.from_euler('xyz', [roll_noise, pitch_noise, yaw_noise])
        quat = r.as_quat()  # [x, y, z, w] 순서
        
        # MuJoCo는 [w, x, y, z] 순서 사용
        self.data.qpos[3] = quat[3]  # w
        self.data.qpos[4] = quat[0]  # x
        self.data.qpos[5] = quat[1]  # y
        self.data.qpos[6] = quat[2]  # z

        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        self.data.qpos[3:7] /= quat_norm

        # 3. 관절 각도 - 다양한 범위의 적응적 노이즈
        joint_noise_scale = noise_scale * 1.2  # 기본 관절 노이즈
        
        # 기본 자연스러운 4족 서있기 관절 각도
        base_joint_targets = np.array([
            # 앞다리 (FR, FL)
            0.0, 0.6, -1.2,    # FR
            0.0, 0.6, -1.2,    # FL
            
            # 뒷다리 (RR, RL)
            0.0, 0.8, -1.5,    # RR
            0.0, 0.8, -1.5     # RL
        ])
        
        # 각 관절마다 다른 범위의 노이즈 적용
        joint_noise = np.zeros(12)
        
        for i in range(12):
            # 관절별로 다른 노이즈 범위 계수 설정
            if i % 3 == 0:  # Hip joints (0, 3, 6, 9)
                range_multiplier = np.random.uniform(0.5, 1.5)
            elif i % 3 == 1:  # Knee joints (1, 4, 7, 10)
                range_multiplier = np.random.uniform(0.8, 2.0)  # 무릎은 더 큰 범위
            else:  # Ankle joints (2, 5, 8, 11)
                range_multiplier = np.random.uniform(0.7, 1.8)  # 발목도 큰 범위
            
            # 초기 훈련 단계에서는 때때로 극단적인 움직임도 시도
            if noise_scale > 0.15 and np.random.random() < 0.3:
                # 30% 확률로 매우 큰 범위 (2~4배)
                extreme_multiplier = np.random.uniform(2.0, 4.0)
                joint_noise[i] = np.random.uniform(-joint_noise_scale * extreme_multiplier, 
                                                   joint_noise_scale * extreme_multiplier)
            else:
                # 일반적인 경우: 관절별 다른 범위
                joint_noise[i] = np.random.uniform(-joint_noise_scale * range_multiplier, 
                                                   joint_noise_scale * range_multiplier)
        
        # 특별히 일부 관절 그룹에 추가 변동
        if noise_scale > 0.1:  # 초기 훈련 단계에서만
            # 앞다리 전체에 코릴레이션된 움직임 추가
            if np.random.random() < 0.4:
                front_leg_bias = np.random.uniform(-joint_noise_scale, joint_noise_scale)
                joint_noise[0:6] += front_leg_bias * 0.5
            
            # 뒷다리 전체에 코릴레이션된 움직임 추가
            if np.random.random() < 0.4:
                rear_leg_bias = np.random.uniform(-joint_noise_scale, joint_noise_scale)
                joint_noise[6:12] += rear_leg_bias * 0.5
            
            # 좌우 비대칭 움직임 추가
            if np.random.random() < 0.3:
                # 왼쪽 다리들에 추가 노이즈
                left_bias = np.random.uniform(-joint_noise_scale * 0.5, joint_noise_scale * 0.5)
                joint_noise[0:3] += left_bias  # FR
                joint_noise[6:9] += left_bias  # RR
                
                # 오른쪽 다리들에 반대 노이즈
                joint_noise[3:6] -= left_bias * 0.7  # FL
                joint_noise[9:12] -= left_bias * 0.7  # RL
        
        # 최종 관절 각도 설정
        joint_targets = base_joint_targets + joint_noise
        
        # 관절 한계 내로 클리핑
        joint_ranges = self.model.jnt_range[1:]  # 첫 번째는 root joint
        for i in range(12):
            joint_targets[i] = np.clip(joint_targets[i], 
                                      joint_ranges[i, 0] * 0.95, 
                                      joint_ranges[i, 1] * 0.95)
        
        self.data.qpos[7:19] = joint_targets

        # 4. 속도에도 다양한 범위의 노이즈 추가
        vel_noise_scale = noise_scale * 0.3
        
        # 속도 노이즈도 다양한 범위로
        for i in range(len(self.data.qvel)):
            if i < 3:  # 선속도
                vel_range = np.random.uniform(0.5, 1.5)
            else:  # 각속도 및 관절 속도
                vel_range = np.random.uniform(0.3, 2.0)
            
            self.data.qvel[i] = np.random.normal(0, vel_noise_scale * vel_range)
        
        self.data.qacc[:] = 0.0

        # 5. 제어 입력 초기화
        self.data.ctrl[:] = 0.0

        # 6. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)

        # 7. 발이 지면에 접촉하도록 높이 자동 조정
        self._auto_adjust_height_for_ground_contact()

    def _auto_adjust_height_for_ground_contact(self):
        """모든 발이 지면에 접촉하도록 로봇 높이 자동 조정"""
        try:
            # 모든 발의 위치 확인
            foot_names = ["FR", "FL", "RR", "RL"]
            foot_positions = []
            
            for foot_name in foot_names:
                try:
                    foot_site_id = self.model.site(foot_name).id
                    foot_pos = self.data.site_xpos[foot_site_id]
                    foot_positions.append(foot_pos[2])  # z 좌표만
                except:
                    continue
            
            if foot_positions:
                # 가장 낮은 발의 z 좌표
                lowest_foot_z = min(foot_positions)
                
                # 지면(z=0)에서 0.5cm 위에 발이 오도록 조정
                target_clearance = 0.005  # 0.5cm
                height_adjustment = target_clearance - lowest_foot_z
                
                # 트렁크 높이 조정
                self.data.qpos[2] += height_adjustment
                
                # 물리 시뮬레이션 재적용
                mujoco.mj_forward(self.model, self.data)
                
        except Exception as e:
            print(f"⚠️ 높이 자동 조정 실패: {e}")

    def reset(self, seed=None, options=None):
        """환경 리셋 - 자연스러운 4족 서있기 자세에서 시작"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # 자연스러운 4족 서있기 자세로 설정
        self._set_natural_standing_pose()

        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0

        return self._get_obs(), info

    def _apply_domain_randomization(self):
        """물리 파라미터 랜덤화"""
        if np.random.random() < 0.7:  # 70% 확률로 적용
            # 중력 변화 (±5%)
            gravity_scale = np.random.uniform(0.95, 1.05)
            self.model.opt.gravity[:] = self.original_gravity * gravity_scale

            # 마찰 변화 (±10%)
            friction_scale = np.random.uniform(0.9, 1.1)
            for i in range(self.model.ngeom):
                if hasattr(self.model, 'geom_friction'):
                    self.model.geom_friction[i, :] *= friction_scale

            # 질량 변화 (±3%)
            mass_scale = np.random.uniform(0.97, 1.03)
            for i in range(self.model.nbody):
                if self.model.body_mass[i] > 0:
                    self.model.body_mass[i] *= mass_scale

    def step(self, action):
        """환경 스텝 실행 - 훈련 진행도 추적 추가"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.standing_reward.compute_reward(self.model, self.data)

        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length

        self.episode_length += 1
        
        # ✅ 훈련 진행도 추적
        self.total_timesteps += 1

        if hasattr(self, 'total_timesteps'):
            self.total_timesteps += 1
        else:
            self.total_timesteps = 1

        info = {
            'episode_length': self.episode_length,
            'standing_reward': reward,
            'standing_success': self._is_standing_successful(),
            'noise_scale': self._get_adaptive_noise_scale(),  # 현재 노이즈 스케일 정보
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _is_terminated(self):
        """2족 보행용 종료 조건"""
        
        # 1. 높이 체크 - 범위 확대
        if self.data.qpos[2] < 0.15 or self.data.qpos[2] > 0.6:
            return True
        
        # 2. 기울기 체크 - 더 관대하게
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.standing_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        # 2족은 더 많은 기울기 허용
        if up_vector[2] < 0.5:  # 60도까지 허용
            return True
        
        # 3. 속도 체크 - 더 관대하게
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        
        # 2족 전환 시 더 많은 움직임 허용
        if linear_vel > 3.0 or angular_vel > 8.0:
            return True
        
        # 4. 안정성 체크 - 연속 불안정만 체크
        if not hasattr(self, '_instability_count'):
            self._instability_count = 0
            
        if self._is_unstable():
            self._instability_count += 1
            if self._instability_count > 50:  # 0.5초 이상 불안정
                return True
        else:
            self._instability_count = 0
        
        return False

    def _is_unstable(self):
        """불안정 상태 판정"""
        # 각속도가 너무 클 때
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        if angular_vel > 4.0:
            return True
        
        # 높이가 너무 낮을 때
        if self.data.qpos[2] < 0.2:
            return True
        
        return False

    def _is_standing_successful(self):
        """4족 서있기 성공 판정"""
        trunk_height = self.data.qpos[2]
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.standing_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]

        # 발 접촉 확인
        foot_contacts = self.standing_reward._get_foot_contacts(self.model, self.data)

        # 성공 조건
        height_ok = 0.25 < trunk_height < 0.38       # 적절한 높이
        upright_ok = up_vector[2] > 0.85             # 충분히 직립
        all_feet_contact = sum(foot_contacts) >= 3.5 # 거의 모든 발이 접촉
        
        # 안정성 조건
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        stable = angular_vel < 1.0 and linear_vel < 0.5

        # 지속 시간 조건
        duration_ok = self.episode_length > 100  # 최소 100 스텝 유지

        return (height_ok and upright_ok and all_feet_contact and 
                stable and duration_ok)

    def _is_foot_contact(self, foot_name):
        """발 접촉 상태 확인"""
        try:
            foot_geom_id = self.model.geom(foot_name).id
            for i in range(self.data.ncon):
                contact_geom1 = self.data.contact[i].geom1
                contact_geom2 = self.data.contact[i].geom2
                if contact_geom1 == foot_geom_id or contact_geom2 == foot_geom_id:
                    contact_force = np.linalg.norm(self.data.contact[i].force)
                    if contact_force > 0.1:
                        return True
            return False
        except:
            return False
    def _get_foot_contacts_direct(self):
        """보상 객체 없이 직접 발 접촉 계산"""
        foot_names = ["FR", "FL", "RR", "RL"]
        contacts = []

        for foot_name in foot_names:
            try:
                foot_geom_id = self.model.geom(foot_name).id
                contact = False

                for i in range(self.data.ncon):
                    contact_geom1 = self.data.contact[i].geom1
                    contact_geom2 = self.data.contact[i].geom2

                    if contact_geom1 == foot_geom_id or contact_geom2 == foot_geom_id:
                        # 접촉력 확인
                        contact_force = np.linalg.norm(self.data.contact[i].force)
                        if contact_force > 0.1:  # 의미있는 접촉
                            contact = True
                            break

                contacts.append(1.0 if contact else 0.0)
            except:
                contacts.append(0.0)

        return contacts

    def _get_reward_object(self):
        """현재 환경의 보상 객체 반환"""
        if hasattr(self, 'bipedal_reward'):
            return self.bipedal_reward
        elif hasattr(self, 'standing_reward'):
            return self.standing_reward
        else:
            return None

class BipedalWalkingEnv(Go1StandingEnv):
    """2족 보행 전용 환경 - 관찰 공간 호환성 개선"""

    def __init__(self, **kwargs):
        # 허용되지 않는 파라미터들 제거
        filtered_kwargs = {}
        allowed_params = {
            'randomize_physics', 'render_mode', 'frame_skip', 
            'observation_space', 'default_camera_config', 'use_base_observation'
        }
        
        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value
        
        # ✅ 호환성 모드 설정
        self._use_base_observation = kwargs.get('use_base_observation', False)
        
        # 부모 클래스 초기화 (필터링된 kwargs 사용)
        super().__init__(**filtered_kwargs)
        
        # 2족 보행용 보상 함수 사용
        self.bipedal_reward = BipedWalkingReward()
        self.episode_length = 0
        self.max_episode_length = 1000

        # 2족 보행을 위한 건강 상태 조건
        self._healthy_z_range = (0.25, 0.60)  # 2족 보행 높이 범위
        self._healthy_pitch_range = (-np.deg2rad(30), np.deg2rad(30))  # 더 관대한 기울기
        self._healthy_roll_range = (-np.deg2rad(30), np.deg2rad(30))

        # Domain randomization 설정
        self.randomize_physics = kwargs.get('randomize_physics', True)
        self.original_gravity = None

        #print(f"🤖 2족 보행 환경 - 관찰 모드: {'기본(45차원)' if self._use_base_observation else '확장(56차원)'}")

    def reset(self, seed=None, options=None):
        """환경 리셋 - 2족 보행 준비 자세에서 시작"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # 2족 보행 준비 자세로 설정
        self._set_bipedal_ready_pose()

        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0

        return self._get_obs(), info

    def step(self, action):
        """환경 스텝 실행"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.bipedal_reward.compute_reward(self.model, self.data)

        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length

        self.episode_length += 1

        if hasattr(self, 'total_timesteps'):
            self.total_timesteps += 1
        else:
            self.total_timesteps = 1

        info = {
            'episode_length': self.episode_length,
            'bipedal_reward': reward,
            'bipedal_success': self._is_bipedal_success(),
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _is_bipedal_success(self):
        """2족 보행 성공 판정"""
        
        # 1. 높이 확인
        trunk_height = self.data.qpos[2]
        height_ok = 0.4 < trunk_height < 0.55
        
        # 2. 앞발이 들려있는지
        front_feet_heights = [
            self._get_foot_height('FR'),
            self._get_foot_height('FL')
        ]
        front_feet_up = all(h > 0.03 for h in front_feet_heights)
        
        # 3. 뒷발만 접촉
        rear_contacts = [
            self._is_foot_contact('RR'),
            self._is_foot_contact('RL')
        ]
        front_contacts = [
            self._is_foot_contact('FR'),
            self._is_foot_contact('FL')
        ]
        rear_feet_only = all(rear_contacts) and not any(front_contacts)
        
        # 4. 안정성
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        stable = angular_vel < 2.0
        
        # 5. 지속 시간
        duration_ok = self.episode_length > 200  # 2초 이상
        
        return (height_ok and front_feet_up and rear_feet_only and 
                stable and duration_ok)


class BipedalCurriculumEnv(BipedalWalkingEnv):
    """2족 보행 커리큘럼 환경"""

    def __init__(self, curriculum_stage=0, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_bipedal_curriculum()

    def _setup_bipedal_curriculum(self):
        """2족 보행 단계별 커리큘럼"""
        
        if self.curriculum_stage == 0:
            # Stage 0: 무게중심 이동 학습
            self.target_height = 0.35
            self.front_feet_target = 0.02  # 살짝만 들기
            self.stability_threshold = 5.0
            
        elif self.curriculum_stage == 1:
            # Stage 1: 앞발 들기
            self.target_height = 0.40
            self.front_feet_target = 0.05
            self.stability_threshold = 4.0
            
        elif self.curriculum_stage == 2:
            # Stage 2: 2족 자세 유지
            self.target_height = 0.45
            self.front_feet_target = 0.08
            self.stability_threshold = 3.0
            
        else:
            # Stage 3+: 안정적 2족 보행
            self.target_height = 0.50
            self.front_feet_target = 0.10
            self.stability_threshold = 2.0

    def advance_curriculum(self, success_rate):
        """성공률에 따라 커리큘럼 진행"""
        if success_rate > 0.80 and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            self._setup_bipedal_curriculum()
            print(f"🎓 2족 보행 커리큘럼 진행: Stage {self.curriculum_stage}")
            return True
        return False


class GradualStandingEnv(Go1StandingEnv):
    """점진적 커리큘럼 4족 서있기 환경"""

    def __init__(self, curriculum_stage=0, **kwargs):
        # 동일한 필터링 적용
        filtered_kwargs = {}
        allowed_params = {
            'randomize_physics', 'render_mode', 'frame_skip', 
            'observation_space', 'default_camera_config', 'use_base_observation'
        }
        
        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value
            elif key != 'curriculum_stage':
                pass
        
        super().__init__(**filtered_kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_curriculum()

    def _setup_curriculum(self):
        """커리큘럼 단계별 설정"""
        if self.curriculum_stage == 0:
            # Stage 0: 기본 균형 유지에 집중
            self.max_episode_length = 500
            self._healthy_z_range = (0.20, 0.42)
            
        elif self.curriculum_stage == 1:
            # Stage 1: 더 정밀한 균형
            self.max_episode_length = 750
            self._healthy_z_range = (0.22, 0.40)
            
        elif self.curriculum_stage == 2:
            # Stage 2: 장시간 유지
            self.max_episode_length = 1000
            self._healthy_z_range = (0.24, 0.38)
            
        else:
            # Stage 3+: 완벽한 서있기
            self.max_episode_length = 1500
            self._healthy_z_range = (0.25, 0.37)

    def advance_curriculum(self, success_rate):
        """성공률에 따라 커리큘럼 진행"""
        if success_rate > 0.80 and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            self._setup_curriculum()
            print(f"🎓 커리큘럼 진행: Stage {self.curriculum_stage}")
            return True
        return False


# ✅ 환경 생성 헬퍼 함수
def create_compatible_env(env_class, pretrained_model_path=None, **env_kwargs):
    """사전훈련 모델과 호환되는 환경 생성"""
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            
            
            # 모델의 관찰 공간 확인
            temp_model = PPO.load(pretrained_model_path, env=None)
            
            if hasattr(temp_model.policy, 'observation_space'):
                model_obs_shape = temp_model.policy.observation_space.shape
            else:
                # 정책 네트워크 크기로 추정
                first_layer = next(temp_model.policy.features_extractor.parameters())
                model_obs_shape = (first_layer.shape[1],)
            
            del temp_model  # 메모리 정리
            
            # 모델이 45차원을 기대하면 호환 모드 사용
            if model_obs_shape[0] == 45:
                env_kwargs['use_base_observation'] = True
                print(f"🔄 호환 모드: 기본 관찰 공간(45차원) 사용")
            else:
                env_kwargs['use_base_observation'] = False
                #print(f"🔄 확장 모드: 2족 보행 관찰 공간({model_obs_shape[0]}차원) 사용")
                
        except Exception as e:
            print(f"⚠️ 모델 분석 실패: {e}, 기본 설정 사용")
            env_kwargs['use_base_observation'] = False
    
    return env_class(**env_kwargs)