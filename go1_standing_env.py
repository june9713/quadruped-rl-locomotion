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
            
            # 동작 관련
            'torso_upright': 8.0,          # 상체 직립
            'smooth_motion': 3.0,          # 부드러운 동작
            'forward_lean': 4.0,           # 적절한 전방 기울기
            
            # 자세 다양성 보상 추가
            'pose_exploration': 2.0,       # 자세 탐색 보상
            'joint_diversity': 1.5,        # 관절 다양성 보상
            
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
        
        # 자세 다양성을 위한 기록
        self.pose_history = deque(maxlen=100)
        self.joint_configurations = deque(maxlen=50)

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
        """2족 보행 보상 계산 - 자세 다양성 보상 추가"""
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
        
        # 5. 상체 직립 보상
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        # 약간의 전방 기울기 허용 (5-15도)
        ideal_pitch = np.deg2rad(10)  # 10도 전방 기울기
        current_pitch = np.arcsin(-trunk_rotation_matrix[2, 0])
        pitch_error = abs(current_pitch - ideal_pitch)
        
        torso_reward = np.exp(-5 * pitch_error) * max(0, up_vector[2])
        total_reward += self.weights['torso_upright'] * torso_reward
        reward_info['torso_upright'] = torso_reward
        
        # 6. 안정성 보상
        angular_vel = data.qvel[3:6]
        angular_stability = np.exp(-2 * np.linalg.norm(angular_vel))
        total_reward += self.weights['angular_stability'] * angular_stability
        reward_info['angular_stability'] = angular_stability
        
        # 7. 자세 다양성 보상 (새로 추가)
        pose_exploration_reward = self._compute_pose_diversity(data)
        total_reward += self.weights['pose_exploration'] * pose_exploration_reward
        reward_info['pose_exploration'] = pose_exploration_reward
        
        # 8. 관절 다양성 보상 (새로 추가)
        joint_diversity_reward = self._compute_joint_diversity(data)
        total_reward += self.weights['joint_diversity'] * joint_diversity_reward
        reward_info['joint_diversity'] = joint_diversity_reward
        
        # 9. 에너지 페널티 (2족은 더 많은 토크 허용)
        motor_efforts = np.sum(np.square(data.ctrl))
        energy_penalty = motor_efforts * 0.5  # 페널티 완화
        total_reward += self.weights['energy'] * energy_penalty
        reward_info['energy'] = -energy_penalty
        
        # 10. 단계별 보너스
        stage_bonus = self._compute_stage_bonus(front_feet_height, rear_feet_contact, trunk_height)
        total_reward += stage_bonus
        reward_info['stage_bonus'] = stage_bonus
        
        return total_reward, reward_info

    def _compute_pose_diversity(self, data):
        """자세 다양성 보상 계산"""
        current_pose = np.concatenate([
            data.qpos[2:7],   # 몸통 위치와 자세
            data.qpos[7:19]   # 관절 각도
        ])
        
        self.pose_history.append(current_pose.copy())
        
        if len(self.pose_history) < 10:
            return 0.0
        
        # 최근 자세들과의 차이 계산
        recent_poses = np.array(list(self.pose_history)[-10:])
        pose_variance = np.var(recent_poses, axis=0)
        diversity_score = np.mean(pose_variance)
        
        # 다양성이 높을수록 보상
        return np.tanh(diversity_score * 5.0)

    def _compute_joint_diversity(self, data):
        """관절 구성 다양성 보상"""
        joint_config = data.qpos[7:19].copy()
        self.joint_configurations.append(joint_config)
        
        if len(self.joint_configurations) < 5:
            return 0.0
        
        # 관절 구성의 표준편차
        configs = np.array(list(self.joint_configurations)[-5:])
        joint_std = np.mean(np.std(configs, axis=0))
        
        # 표준편차가 클수록 다양성이 높음
        return np.tanh(joint_std * 2.0)

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

    def _compute_stage_bonus(self, front_feet_height, rear_feet_contact, trunk_height):
        """단계별 보너스 계산"""
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
        
        # 2족 서기 단계: 모든 조건 만족
        if (trunk_height > 0.4 and 
            np.mean(front_feet_height) > 0.05 and 
            np.mean(rear_feet_contact) > 0.9):
            stage_bonus += 5.0
        
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

        # 🎯 다양한 자세를 위한 추가 설정
        self.pose_templates = self._create_pose_templates()
        self.joint_noise_scales = self._create_joint_noise_scales()
        self.exploration_bonus = 0.0
        self.pose_diversity_counter = 0

    def _create_pose_templates(self):
        """다양한 시작 자세 템플릿 생성"""
        templates = {
            'normal_standing': {
                'name': '일반 서기',
                'front_joints': [0.0, 0.6, -1.2, 0.0, 0.6, -1.2],
                'rear_joints': [0.0, 0.8, -1.5, 0.0, 0.8, -1.5],
                'height': 0.30,
                'pitch': 0.0
            },
            'crouched': {
                'name': '웅크린 자세',
                'front_joints': [0.2, 0.9, -1.8, -0.2, 0.9, -1.8],
                'rear_joints': [0.1, 1.2, -2.0, -0.1, 1.2, -2.0],
                'height': 0.25,
                'pitch': np.deg2rad(-5)
            },
            'stretched': {
                'name': '쭉 편 자세',
                'front_joints': [-0.2, 0.3, -0.6, 0.2, 0.3, -0.6],
                'rear_joints': [-0.1, 0.5, -1.0, 0.1, 0.5, -1.0],
                'height': 0.35,
                'pitch': np.deg2rad(3)
            },
            'asymmetric_1': {
                'name': '비대칭 자세 1',
                'front_joints': [0.3, 0.7, -1.4, -0.2, 0.5, -1.0],
                'rear_joints': [0.2, 0.9, -1.7, -0.3, 1.1, -1.9],
                'height': 0.28,
                'pitch': np.deg2rad(-2)
            },
            'asymmetric_2': {
                'name': '비대칭 자세 2',
                'front_joints': [-0.1, 0.8, -1.6, 0.4, 0.4, -0.8],
                'rear_joints': [-0.2, 1.0, -1.8, 0.3, 0.7, -1.3],
                'height': 0.32,
                'pitch': np.deg2rad(4)
            },
            'wide_stance': {
                'name': '넓은 자세',
                'front_joints': [0.5, 0.6, -1.2, -0.5, 0.6, -1.2],
                'rear_joints': [0.4, 0.8, -1.5, -0.4, 0.8, -1.5],
                'height': 0.29,
                'pitch': 0.0
            },
            'narrow_stance': {
                'name': '좁은 자세',
                'front_joints': [-0.1, 0.7, -1.4, 0.1, 0.7, -1.4],
                'rear_joints': [-0.05, 0.9, -1.6, 0.05, 0.9, -1.6],
                'height': 0.31,
                'pitch': 0.0
            },
            'ready_to_bipedal': {
                'name': '2족 준비 자세',
               'front_joints': [0.1, 0.3, -0.6, -0.1, 0.3, -0.6],
               'rear_joints': [0.0, 0.4, -0.8, 0.0, 0.4, -0.8],
               'height': 0.35,
               'pitch': np.deg2rad(-5)
            },
           'dynamic_1': {
               'name': '동적 자세 1',
               'front_joints': [0.6, 0.2, -0.4, -0.3, 0.8, -1.6],
               'rear_joints': [0.3, 0.6, -1.2, -0.2, 1.0, -1.8],
               'height': 0.33,
               'pitch': np.deg2rad(7)
            },
           'dynamic_2': {
               'name': '동적 자세 2',
               'front_joints': [-0.4, 0.9, -1.8, 0.5, 0.1, -0.2],
               'rear_joints': [-0.1, 1.3, -2.2, 0.2, 0.5, -1.0],
               'height': 0.27,
               'pitch': np.deg2rad(-8)
            }
        }
        return templates

    def _create_joint_noise_scales(self):
        """관절별 노이즈 스케일 생성 - 각 관절마다 다른 노이즈"""
        # 관절 순서: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, 
        #            RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        return {
            'position_noise': np.array([
                0.15,  # FR_hip - 큰 노이즈
                0.12,  # FR_thigh
                0.18,  # FR_calf - 가장 큰 노이즈
                0.15,  # FL_hip
                0.12,  # FL_thigh  
                0.18,  # FL_calf
                0.10,  # RR_hip - 상대적으로 작은 노이즈
                0.08,  # RR_thigh
                0.14,  # RR_calf
                0.10,  # RL_hip
                0.08,  # RL_thigh
                0.14   # RL_calf
            ]),
            'velocity_noise': np.array([
                0.08, 0.06, 0.10,  # 앞다리
                0.08, 0.06, 0.10,
                0.05, 0.04, 0.07,  # 뒷다리 - 더 안정적
                0.05, 0.04, 0.07
            ]),
            'height_variation': np.array([
                0.02, 0.03, 0.01,  # 앞다리 높이 변화
                0.02, 0.03, 0.01,
                0.01, 0.02, 0.01,  # 뒷다리
                0.01, 0.02, 0.01
            ])
        }

    def _get_base_obs(self):
        """기본 Go1MujocoEnv와 호환되는 관찰 상태 (45차원)"""
        # 부모 클래스의 관찰 방법 사용
        return super()._get_obs()
    
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

    def _set_diverse_bipedal_pose(self):
        """🎯 다양한 2족 보행 준비 자세 설정"""
        
        # 랜덤하게 자세 템플릿 선택
        template_name = np.random.choice(list(self.pose_templates.keys()))
        template = self.pose_templates[template_name]
        
        print(f"🎭 선택된 자세: {template['name']}")
        
        # 1. 트렁크 위치 - 템플릿 기반 + 추가 노이즈
        base_height = template['height']
        height_noise = np.random.uniform(-0.03, 0.03)
        self.data.qpos[0] = np.random.uniform(-0.02, 0.02)  # x: 더 큰 변동
        self.data.qpos[1] = np.random.uniform(-0.02, 0.02)  # y: 더 큰 변동  
        self.data.qpos[2] = base_height + height_noise       # z: 템플릿 기반

        # 2. 트렁크 자세 - 템플릿의 pitch + 추가 변동
        base_pitch = template['pitch']
        pitch_noise = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        roll_noise = np.random.uniform(-np.deg2rad(5), np.deg2rad(5))
        yaw_noise = np.random.uniform(-np.deg2rad(8), np.deg2rad(8))
        
        final_pitch = base_pitch + pitch_noise
        
        # 오일러 각을 쿼터니언으로 변환
        quat = self._euler_to_quat(roll_noise, final_pitch, yaw_noise)
        self.data.qpos[3:7] = quat

        # 3. 관절 각도 - 템플릿 + 개별 노이즈
        joint_targets = np.array(template['front_joints'] + template['rear_joints'])
        
        # 🎯 관절별 개별 노이즈 적용
        position_noise = np.random.normal(0, self.joint_noise_scales['position_noise'])
        
        # 추가 다양성을 위한 랜덤 변동
        if np.random.random() < 0.3:  # 30% 확률로 극단적 변동
            extreme_noise = np.random.normal(0, 0.25, 12) * np.random.choice([-1, 1], 12)
            position_noise += extreme_noise * 0.5
        
        # 관절 대칭성을 때때로 의도적으로 깨뜨리기
        if np.random.random() < 0.4:  # 40% 확률로 비대칭
            # 좌우 다리에 다른 노이즈 적용
            asymmetry_factor = np.random.uniform(0.5, 1.5)
            position_noise[0:3] *= asymmetry_factor  # FR
            position_noise[6:9] *= asymmetry_factor  # RR
            position_noise[3:6] *= (2.0 - asymmetry_factor)  # FL
            position_noise[9:12] *= (2.0 - asymmetry_factor)  # RL
        
        joint_targets += position_noise
        
        # 관절 위치 설정
        self.data.qpos[7:19] = joint_targets

        # 4. 초기 속도 - 다양성을 위한 작은 초기 속도
        self.data.qvel[:3] = np.random.normal(0, 0.05, 3)  # 작은 선형 속도
        self.data.qvel[3:6] = np.random.normal(0, 0.1, 3)  # 작은 각속도
        
        # 관절 속도에도 노이즈 추가
        joint_vel_noise = np.random.normal(0, self.joint_noise_scales['velocity_noise'])
        self.data.qvel[7:] = joint_vel_noise

        # 5. 제어 입력에도 초기 노이즈
        control_noise = np.random.normal(0, 0.1, 12)
        self.data.ctrl[:] = control_noise

        # 6. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)

        # 7. 자동 높이 조정 (선택적으로)
        if np.random.random() < 0.7:  # 70% 확률로만 자동 조정
            self._auto_adjust_height_for_ground_contact()
        
        # 8. 자세 다양성 추적
        self.pose_diversity_counter += 1
        if hasattr(self, 'used_poses'):
            self.used_poses.append(template_name)
        else:
            self.used_poses = [template_name]

    def _euler_to_quat(self, roll, pitch, yaw):
        """오일러 각을 쿼터니언으로 변환"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def _generate_random_pose(self):
        """🎲 완전히 랜덤한 자세 생성"""
        print("🎲 완전 랜덤 자세 생성")
        
        # 1. 랜덤 높이
        self.data.qpos[2] = np.random.uniform(0.25, 0.40)
        
        # 2. 랜덤 트렁크 자세
        roll = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
        pitch = np.random.uniform(-np.deg2rad(20), np.deg2rad(20))
        yaw = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        quat = self._euler_to_quat(roll, pitch, yaw)
        self.data.qpos[3:7] = quat
        
        # 3. 완전 랜덤 관절 각도 (관절 한계 내에서)
        joint_ranges = self.model.jnt_range[1:13]  # 관절 범위
        random_joints = []
        
        for i, (min_angle, max_angle) in enumerate(joint_ranges):
            # 관절별로 다른 랜덤 범위 사용
            range_factor = np.random.uniform(0.6, 1.0)  # 60-100% 범위 사용
            center = (min_angle + max_angle) / 2
            half_range = (max_angle - min_angle) / 2 * range_factor
            
            random_angle = np.random.uniform(
                center - half_range, 
                center + half_range
            )
            random_joints.append(random_angle)
        
        self.data.qpos[7:19] = np.array(random_joints)
        
        # 4. 랜덤 초기 속도
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # 물리 적용
        mujoco.mj_forward(self.model, self.data)

    def _set_bipedal_ready_pose(self):
        """2족 보행 준비 자세 설정 - 다양성 대폭 강화"""
        
        # 🎯 다양한 자세 전략
        pose_strategy = np.random.choice([
            'template_based',    # 50% - 템플릿 기반
            'random_generated',  # 30% - 완전 랜덤
            'hybrid'            # 20% - 하이브리드
        ], p=[0.5, 0.3, 0.2])
        
        if pose_strategy == 'template_based':
            self._set_diverse_bipedal_pose()
        elif pose_strategy == 'random_generated':
            self._generate_random_pose()
        else:  # hybrid
            # 템플릿 시작 후 큰 노이즈 추가
            self._set_diverse_bipedal_pose()
            
            # 추가 하이브리드 변동
            hybrid_noise = np.random.normal(0, 0.2, 12)
            self.data.qpos[7:19] += hybrid_noise
            
            # 높이도 추가 변동
            self.data.qpos[2] += np.random.uniform(-0.05, 0.05)
            
            print("🔀 하이브리드 자세 적용")
        
        # 최종 안전성 검사 및 조정
        self._ensure_pose_safety()

    def _ensure_pose_safety(self):
        """자세 안전성 보장"""
        # 높이 제한
        self.data.qpos[2] = np.clip(self.data.qpos[2], 0.20, 0.50)
        
        # 관절 한계 검사
        joint_ranges = self.model.jnt_range[1:13]
        for i, (min_angle, max_angle) in enumerate(joint_ranges):
            self.data.qpos[7+i] = np.clip(self.data.qpos[7+i], min_angle, max_angle)
        
        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        if quat_norm > 0:
            self.data.qpos[3:7] /= quat_norm

    def _set_natural_standing_pose(self):
        """자연스러운 4족 서있기 자세 설정 - 다양성 강화"""
        
        # 🎯 4족 자세도 다양하게
        stance_type = np.random.choice([
            'normal', 'wide', 'narrow', 'asymmetric', 'crouched', 'alert'
        ])
        
        # 1. 트렁크 위치 설정
        self.data.qpos[0] = np.random.uniform(-0.02, 0.02)
        self.data.qpos[1] = np.random.uniform(-0.02, 0.02)
        
        if stance_type == 'normal':
            base_height = 0.30
            joint_base = [0.0, 0.6, -1.2, 0.0, 0.6, -1.2, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5]
        elif stance_type == 'wide':
            base_height = 0.28
            joint_base = [0.4, 0.6, -1.2, -0.4, 0.6, -1.2, 0.3, 0.8, -1.5, -0.3, 0.8, -1.5]
        elif stance_type == 'narrow':
            base_height = 0.32
            joint_base = [-0.1, 0.7, -1.4, 0.1, 0.7, -1.4, -0.05, 0.9, -1.6, 0.05, 0.9, -1.6]
        elif stance_type == 'asymmetric':
            base_height = 0.29
            joint_base = [0.2, 0.5, -1.0, -0.3, 0.8, -1.6, 0.1, 0.7, -1.3, -0.2, 1.0, -1.8]
        elif stance_type == 'crouched':
            base_height = 0.25
            joint_base = [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 1.2, -2.0, 0.0, 1.2, -2.0]
        else:  # alert
            base_height = 0.33
            joint_base = [0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, 0.6, -1.2, 0.0, 0.6, -1.2]
        
        self.data.qpos[2] = base_height + np.random.uniform(-0.02, 0.02)

        # 2. 트렁크 자세 (다양한 기울기)
        roll = np.random.uniform(-np.deg2rad(5), np.deg2rad(5))
        pitch = np.random.uniform(-np.deg2rad(8), np.deg2rad(8))
        yaw = np.random.uniform(-np.deg2rad(5), np.deg2rad(5))
        quat = self._euler_to_quat(roll, pitch, yaw)
        self.data.qpos[3:7] = quat

        # 3. 관절 각도 - 큰 노이즈 적용
        joint_targets = np.array(joint_base)
        
        # 관절별 개별 노이즈
        large_noise = np.random.normal(0, self.joint_noise_scales['position_noise'] * 1.5)
        joint_targets += large_noise
        
        self.data.qpos[7:19] = joint_targets

        # 4. 초기 속도
        self.data.qvel[:] = np.random.normal(0, 0.08, len(self.data.qvel))

        # 5. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)
        
        # 6. 높이 조정 (50% 확률)
        if np.random.random() < 0.5:
            self._auto_adjust_height_for_ground_contact()
        
        print(f"🐕 4족 {stance_type} 자세 설정")

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
                
                # 지면(z=0)에서 1cm 위에 발이 오도록 조정 (기존 0.5cm에서 증가)
                target_clearance = np.random.uniform(0.005, 0.015)  # 0.5-1.5cm 랜덤
                height_adjustment = target_clearance - lowest_foot_z
                
                # 트렁크 높이 조정
                self.data.qpos[2] += height_adjustment
                
                # 물리 시뮬레이션 재적용
                mujoco.mj_forward(self.model, self.data)
                
        except Exception as e:
            print(f"⚠️ 높이 자동 조정 실패: {e}")

    def reset(self, seed=None, options=None):
        """환경 리셋 - 다양한 자세에서 시작"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # 🎯 자세 선택 - 더 공격적인 다양성
        pose_choice = np.random.choice([
            'natural_standing',  # 40% - 4족 서기  
            'bipedal_ready'      # 60% - 2족 준비
        ], p=[0.4, 0.6])
        
        if pose_choice == 'natural_standing':
            self._set_natural_standing_pose()
        else:
            self._set_bipedal_ready_pose()

        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0
        
        # 다양성 통계 업데이트
        if self.episode_length == 0:  # 새 에피소드 시작
            self._update_diversity_stats()

        return self._get_obs(), info

    def _update_diversity_stats(self):
        """다양성 통계 업데이트"""
        if hasattr(self, 'used_poses') and len(self.used_poses) > 0:
            unique_poses = len(set(self.used_poses[-50:]))  # 최근 50개 에피소드
            diversity_ratio = unique_poses / min(50, len(self.used_poses))
            
            if diversity_ratio > 0.7:
                self.exploration_bonus = 1.0
                if len(self.used_poses) % 20 == 0:  # 20 에피소드마다
                    print(f"🎭 자세 다양성: {diversity_ratio:.1%} ({unique_poses}/50)")
            else:
                self.exploration_bonus = 0.5

    def _apply_domain_randomization(self):
        """물리 파라미터 랜덤화 - 더 강화"""
        if np.random.random() < 0.8:  # 80% 확률로 적용 (기존 70%에서 증가)
            # 1. 중력 변화 (±10%)
            gravity_scale = np.random.uniform(0.9, 1.1)
            self.model.opt.gravity[:] = self.original_gravity * gravity_scale

            # 2. 마찰 변화 (±20%)
            friction_scale = np.random.uniform(0.8, 1.2)
            for i in range(self.model.ngeom):
                if hasattr(self.model, 'geom_friction'):
                    self.model.geom_friction[i, :] *= friction_scale

            # 3. 질량 변화 (±8%)
            mass_scale = np.random.uniform(0.92, 1.08)
            for i in range(self.model.nbody):
                if self.model.body_mass[i] > 0:
                    self.model.body_mass[i] *= mass_scale
            
            # 4. 관절 강성 변화 (새로 추가)
            if np.random.random() < 0.5:
                stiffness_scale = np.random.uniform(0.8, 1.2)
                if hasattr(self.model, 'jnt_stiffness'):
                    self.model.jnt_stiffness[1:] *= stiffness_scale
            
            # 5. 댐핑 변화 (새로 추가)
            if np.random.random() < 0.4:
                damping_scale = np.random.uniform(0.7, 1.3)
                if hasattr(self.model, 'dof_damping'):
                    self.model.dof_damping[6:] *= damping_scale

    def step(self, action):
        """환경 스텝 실행"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.standing_reward.compute_reward(self.model, self.data)
        
        # 🎯 다양성 보너스 추가
        if hasattr(self, 'exploration_bonus'):
            reward += self.exploration_bonus * 0.1

        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length

        self.episode_length += 1

        if hasattr(self, 'total_timesteps'):
            self.total_timesteps += 1
        else:
            self.total_timesteps = 1

        info = {
            'episode_length': self.episode_length,
            'standing_reward': reward,
            'standing_success': self._is_standing_successful(),
            'exploration_bonus': getattr(self, 'exploration_bonus', 0.0),
            'pose_diversity': len(set(getattr(self, 'used_poses', []))) if hasattr(self, 'used_poses') else 0,
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
    """2족 보행 전용 환경 - 관찰 공간 호환성 개선 + 다양성 강화"""

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

        # 🎯 2족 보행 전용 다양성 강화
        self._create_bipedal_pose_templates()
        self.bipedal_joint_noise_scales = self._create_bipedal_joint_noise_scales()
        
        print(f"🤖 2족 보행 환경 - 다양성 강화 모드 활성화")

    def _create_bipedal_pose_templates(self):
        """2족 보행 전용 자세 템플릿"""
        self.bipedal_pose_templates = {
            'conservative_bipedal': {
                'name': '보수적 2족',
                'front_joints': [0.0, 0.2, -0.4, 0.0, 0.2, -0.4],
                'rear_joints': [0.0, 0.3, -0.6, 0.0, 0.3, -0.6],
                'height': 0.38,
                'pitch': np.deg2rad(-3)
            },
            'aggressive_bipedal': {
                'name': '공격적 2족',
                'front_joints': [0.2, 0.1, -0.2, -0.2, 0.1, -0.2],
                'rear_joints': [0.0, 0.5, -1.0, 0.0, 0.5, -1.0],
                'height': 0.45,
                'pitch': np.deg2rad(-8)
            },
            'extreme_lift': {
                'name': '극단적 들기',
                'front_joints': [0.5, -0.2, 0.4, -0.5, -0.2, 0.4],
                'rear_joints': [0.0, 0.8, -1.6, 0.0, 0.8, -1.6],
                'height': 0.42,
                'pitch': np.deg2rad(-12)
            },
            'asymmetric_bipedal': {
                'name': '비대칭 2족',
                'front_joints': [0.3, 0.0, -0.1, -0.1, 0.4, -0.8],
                'rear_joints': [0.1, 0.4, -0.8, -0.2, 0.6, -1.2],
                'height': 0.40,
                'pitch': np.deg2rad(-5)
            },
            'wide_bipedal': {
                'name': '넓은 2족',
                'front_joints': [0.6, 0.1, -0.2, -0.6, 0.1, -0.2],
                'rear_joints': [0.4, 0.4, -0.8, -0.4, 0.4, -0.8],
                'height': 0.41,
                'pitch': np.deg2rad(-6)
            },
            'crouched_bipedal': {
                'name': '웅크린 2족',
                'front_joints': [0.0, 0.6, -1.2, 0.0, 0.6, -1.2],
                'rear_joints': [0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
                'height': 0.35,
                'pitch': np.deg2rad(0)
            },
            'dancing_bipedal': {
                'name': '춤추는 2족',
                'front_joints': [0.8, -0.5, 1.0, -0.3, 0.2, -0.4],
                'rear_joints': [0.2, 0.6, -1.2, -0.1, 0.5, -1.0],
                'height': 0.43,
                'pitch': np.deg2rad(-10)
            }
        }

    def _create_bipedal_joint_noise_scales(self):
        """2족 보행용 관절 노이즈 스케일 - 더 공격적"""
        return {
            'position_noise': np.array([
                0.25,  # FR_hip - 앞다리는 더 큰 노이즈
                0.20,  # FR_thigh
                0.30,  # FR_calf - 가장 큰 노이즈
                0.25,  # FL_hip
                0.20,  # FL_thigh  
                0.30,  # FL_calf
                0.08,  # RR_hip - 뒷다리는 상대적으로 안정
                0.06,  # RR_thigh
                0.12,  # RR_calf
                0.08,  # RL_hip
                0.06,  # RL_thigh
                0.12   # RL_calf
            ]),
            'velocity_noise': np.array([
                0.15, 0.12, 0.20,  # 앞다리 - 큰 속도 변동
                0.15, 0.12, 0.20,
                0.04, 0.03, 0.06,  # 뒷다리 - 안정적
                0.04, 0.03, 0.06
            ]),
            'extreme_noise': np.array([
                0.4, 0.3, 0.5,  # 극단적 변동용
                0.4, 0.3, 0.5,
                0.1, 0.08, 0.15,
                0.1, 0.08, 0.15
            ])
        }

    def _set_extreme_bipedal_pose(self):
        """🚀 극단적 2족 보행 자세 설정"""
        
        # 극단적 자세 템플릿 선택
        extreme_templates = ['aggressive_bipedal', 'extreme_lift', 'dancing_bipedal']
        template_name = np.random.choice(extreme_templates)
        template = self.bipedal_pose_templates[template_name]
        
        print(f"🚀 극단적 자세: {template['name']}")
        
        # 기본 설정
        self.data.qpos[0] = np.random.uniform(-0.03, 0.03)
        self.data.qpos[1] = np.random.uniform(-0.03, 0.03)  
        self.data.qpos[2] = template['height'] + np.random.uniform(-0.05, 0.05)

        # 극단적 기울기
        base_pitch = template['pitch']
        pitch_noise = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
        roll_noise = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        yaw_noise = np.random.uniform(-np.deg2rad(12), np.deg2rad(12))
        
        quat = self._euler_to_quat(roll_noise, base_pitch + pitch_noise, yaw_noise)
        self.data.qpos[3:7] = quat

        # 관절 각도 - 극단적 노이즈
        joint_targets = np.array(template['front_joints'] + template['rear_joints'])
        extreme_noise = np.random.normal(0, self.bipedal_joint_noise_scales['extreme_noise'])
        
        # 50% 확률로 완전 비대칭 적용
        if np.random.random() < 0.5:
            asymmetry = np.random.uniform(0.3, 2.0)
            extreme_noise[0:6] *= asymmetry  # 앞다리
            extreme_noise[6:12] *= (2.5 - asymmetry)  # 뒷다리
        
        joint_targets += extreme_noise
        self.data.qpos[7:19] = joint_targets

        # 극단적 초기 속도
        self.data.qvel[:3] = np.random.normal(0, 0.1, 3)
        self.data.qvel[3:6] = np.random.normal(0, 0.2, 3)
        self.data.qvel[7:] = np.random.normal(0, 0.15, 12)

        # 물리 적용
        mujoco.mj_forward(self.model, self.data)
        self._ensure_pose_safety()

    def _set_diverse_bipedal_pose(self):
        """다양한 2족 보행 자세 설정 - 오버라이드"""
        
        # 2족 보행 전용 템플릿 사용
        template_name = np.random.choice(list(self.bipedal_pose_templates.keys()))
        template = self.bipedal_pose_templates[template_name]
        
        print(f"🎭 2족 자세: {template['name']}")
        
        # 위치 설정
        self.data.qpos[0] = np.random.uniform(-0.02, 0.02)
        self.data.qpos[1] = np.random.uniform(-0.02, 0.02)  
        self.data.qpos[2] = template['height'] + np.random.uniform(-0.03, 0.03)

        # 자세 설정
        base_pitch = template['pitch']
        pitch_noise = np.random.uniform(-np.deg2rad(12), np.deg2rad(12))
        roll_noise = np.random.uniform(-np.deg2rad(8), np.deg2rad(8))
        yaw_noise = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        
        quat = self._euler_to_quat(roll_noise, base_pitch + pitch_noise, yaw_noise)
        self.data.qpos[3:7] = quat

        # 관절 설정
        joint_targets = np.array(template['front_joints'] + template['rear_joints'])
        position_noise = np.random.normal(0, self.bipedal_joint_noise_scales['position_noise'])
        
        # 2족 특화 비대칭 적용
        if np.random.random() < 0.6:  # 60% 확률
            front_asymmetry = np.random.uniform(0.4, 1.6)
            position_noise[0:3] *= front_asymmetry
            position_noise[3:6] *= (2.0 - front_asymmetry)
        
        joint_targets += position_noise
        self.data.qpos[7:19] = joint_targets

        # 속도 설정
        self.data.qvel[:] = np.random.normal(0, 0.1, len(self.data.qvel))
        
        # 물리 적용
        mujoco.mj_forward(self.model, self.data)

    def _set_bipedal_ready_pose(self):
        """2족 보행 준비 자세 - 더욱 다양하게"""
        
        # 🎯 2족 보행 전용 다양성 전략
        bipedal_strategy = np.random.choice([
            'template_based',     # 40%
            'extreme_pose',       # 30%
            'random_generated',   # 20%
            'hybrid_chaos'        # 10%
        ], p=[0.4, 0.3, 0.2, 0.1])
        
        if bipedal_strategy == 'template_based':
            self._set_diverse_bipedal_pose()
        elif bipedal_strategy == 'extreme_pose':
            self._set_extreme_bipedal_pose()
        elif bipedal_strategy == 'random_generated':
            self._generate_random_bipedal_pose()
        else:  # hybrid_chaos
            # 모든 전략 조합
            self._set_diverse_bipedal_pose()
            chaos_noise = np.random.normal(0, 0.3, 12)
            self.data.qpos[7:19] += chaos_noise
            self.data.qpos[2] += np.random.uniform(-0.08, 0.08)
            print("🌪️ 카오스 하이브리드 자세")
        
        self._ensure_pose_safety()

    def _generate_random_bipedal_pose(self):
        """완전 랜덤 2족 보행 자세"""
        print("🎲 랜덤 2족 자세 생성")
        
        # 2족에 적합한 높이
        self.data.qpos[2] = np.random.uniform(0.32, 0.50)
        
        # 2족 기울기
        roll = np.random.uniform(-np.deg2rad(20), np.deg2rad(20))
        pitch = np.random.uniform(-np.deg2rad(25), np.deg2rad(10))
        yaw = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))
        quat = self._euler_to_quat(roll, pitch, yaw)
        self.data.qpos[3:7] = quat
        
        # 2족 특화 관절 각도
        joint_ranges = self.model.jnt_range[1:13]
        random_joints = []
        
        for i, (min_angle, max_angle) in enumerate(joint_ranges):
            if i < 6:  # 앞다리 - 더 자유롭게
                range_factor = np.random.uniform(0.4, 1.0)
            else:  # 뒷다리 - 안정적으로
                range_factor = np.random.uniform(0.6, 0.9)
                
            center = (min_angle + max_angle) / 2
            half_range = (max_angle - min_angle) / 2 * range_factor
            
            random_angle = np.random.uniform(center - half_range, center + half_range)
            random_joints.append(random_angle)
        
        self.data.qpos[7:19] = np.array(random_joints)
        
        # 랜덤 속도
        self.data.qvel[:] = np.random.normal(0, 0.12, len(self.data.qvel))
        
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        """환경 리셋 - 2족 보행 전용 다양성"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # 🎯 2족 보행만 시도 (더 다양하게)
        self._set_bipedal_ready_pose()

        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0
        self._update_diversity_stats()

        return self._get_obs(), info

    def step(self, action):
        """환경 스텝 실행"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.bipedal_reward.compute_reward(self.model, self.data)
        
        # 다양성 보너스 강화
        if hasattr(self, 'exploration_bonus'):
            reward += self.exploration_bonus * 0.2  # 기존 0.1에서 0.2로 증가

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
            'exploration_bonus': getattr(self, 'exploration_bonus', 0.0),
            'pose_diversity': len(set(getattr(self, 'used_poses', []))) if hasattr(self, 'used_poses') else 0,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _is_bipedal_success(self):
        """2족 보행 성공 판정"""
        
        # 1. 높이 확인
        trunk_height = self.data.qpos[2]
        height_ok = 0.35 < trunk_height < 0.55  # 범위 확대
        
        # 2. 앞발이 들려있는지
        front_feet_heights = [
            self._get_foot_height('FR'),
            self._get_foot_height('FL')
        ]
        front_feet_up = all(h > 0.02 for h in front_feet_heights)  # 기준 완화
        
        # 3. 뒷발만 접촉
        rear_contacts = [
            self._is_foot_contact('RR'),
            self._is_foot_contact('RL')
        ]
        front_contacts = [
            self._is_foot_contact('FR'),
            self._is_foot_contact('FL')
        ]
        rear_feet_only = sum(rear_contacts) >= 1 and sum(front_contacts) == 0  # 조건 완화
        
        # 4. 안정성
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        stable = angular_vel < 3.0  # 기준 완화
        
        # 5. 지속 시간
        duration_ok = self.episode_length > 150  # 1.5초 이상
        
        return (height_ok and front_feet_up and rear_feet_only and 
                stable and duration_ok)


class BipedalCurriculumEnv(BipedalWalkingEnv):
    """2족 보행 커리큘럼 환경 - 다양성 강화"""

    def __init__(self, curriculum_stage=0, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_bipedal_curriculum()

    def _setup_bipedal_curriculum(self):
        """2족 보행 단계별 커리큘럼"""
        
        if self.curriculum_stage == 0:
            # Stage 0: 기본 다양성 + 무게중심 이동
            self.target_height = 0.35
            self.front_feet_target = 0.02
            self.stability_threshold = 5.0
            self.diversity_weight = 1.0
            
        elif self.curriculum_stage == 1:
            # Stage 1: 중간 다양성 + 앞발 들기
            self.target_height = 0.40
            self.front_feet_target = 0.05
            self.stability_threshold = 4.0
            self.diversity_weight = 1.5
            
        elif self.curriculum_stage == 2:
            # Stage 2: 높은 다양성 + 2족 자세 유지
            self.target_height = 0.45
            self.front_feet_target = 0.08
            self.stability_threshold = 3.0
            self.diversity_weight = 2.0
            
        else:
            # Stage 3+: 최대 다양성 + 안정적 2족 보행
            self.target_height = 0.50
            self.front_feet_target = 0.10
            self.stability_threshold = 2.0
            self.diversity_weight = 2.5
        
        # 단계별 노이즈 스케일 조정
        noise_scale = 1.0 + (self.curriculum_stage * 0.2)
        for key in self.bipedal_joint_noise_scales:
            self.bipedal_joint_noise_scales[key] *= noise_scale

    def advance_curriculum(self, success_rate):
        """성공률에 따라 커리큘럼 진행"""
        if success_rate > 0.80 and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            self._setup_bipedal_curriculum()
            print(f"🎓 2족 보행 커리큘럼 진행: Stage {self.curriculum_stage} (다양성 가중치: {self.diversity_weight})")
            return True
        return False


class GradualStandingEnv(Go1StandingEnv):
    """점진적 커리큘럼 4족 서있기 환경 - 다양성 강화"""

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
        """커리큘럼 단계별 설정 - 다양성 점진적 증가"""
        if self.curriculum_stage == 0:
            # Stage 0: 기본 균형 + 제한된 다양성
            self.max_episode_length = 500
            self._healthy_z_range = (0.20, 0.42)
            self.diversity_factor = 0.5
            
        elif self.curriculum_stage == 1:
            # Stage 1: 더 정밀한 균형 + 중간 다양성
            self.max_episode_length = 750
            self._healthy_z_range = (0.22, 0.40)
            self.diversity_factor = 0.8
            
        elif self.curriculum_stage == 2:
            # Stage 2: 장시간 유지 + 높은 다양성
            self.max_episode_length = 1000
            self._healthy_z_range = (0.24, 0.38)
            self.diversity_factor = 1.2
            
        else:
            # Stage 3+: 완벽한 서있기 + 최대 다양성
            self.max_episode_length = 1500
            self._healthy_z_range = (0.25, 0.37)
            self.diversity_factor = 1.5
        
        # 단계별 노이즈 스케일 조정
        for key in self.joint_noise_scales:
            self.joint_noise_scales[key] *= self.diversity_factor

    def advance_curriculum(self, success_rate):
        """성공률에 따라 커리큘럼 진행"""
        if success_rate > 0.80 and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            self._setup_curriculum()
            print(f"🎓 커리큘럼 진행: Stage {self.curriculum_stage} (다양성 계수: {self.diversity_factor})")
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