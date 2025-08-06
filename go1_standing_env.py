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


class RobotPhysicsUtils:
    """로봇 물리 계산을 위한 공통 유틸리티 클래스"""

    GLOBAL_RANDOMNESS_INTENSITY = 0.18  # 기본값 1.0 (0.0 = 랜덤성 없음, 2.0 = 2배 강화)

    # 공통 관절 각도 상수들
    NATURAL_STANDING_JOINTS = np.array([
        # 앞다리 (FR, FL) - 자연스러운 4족 서기
        0.0, 0.6, -1.2,    # FR
        0.0, 0.6, -1.2,    # FL
        # 뒷다리 (RR, RL)
        0.0, 0.8, -1.5,    # RR
        0.0, 0.8, -1.5     # RL
    ])
    
    BIPEDAL_READY_JOINTS = np.array([
        # 앞다리 (FR, FL) - 몸쪽으로 당긴 상태
        0.0, 1.0, -0.6,    # FR
        0.0, 1.0, -0.0,    # FL
        # 뒷다리 (RR, RL) - 몸을 지지하기 좋게 굽힌 상태  
        0.0, 1.0, 0.0,     # RR
        0.0, 2.0, 0.0,     # RL
    ])

    @classmethod
    def set_randomness_intensity(cls, intensity):
        """
        전역 랜덤성 강도 설정
        
        Args:
            intensity (float): 랜덤성 강도
                - 0.0: 완전히 랜덤성 없음 (항상 동일한 초기 자세)
                - 0.5: 약한 랜덤성
                - 1.0: 기본 랜덤성 (기본값)
                - 2.0: 강한 랜덤성
                - 3.0: 매우 강한 랜덤성
        """
        cls.GLOBAL_RANDOMNESS_INTENSITY = max(0.0, intensity)  # 음수 방지
        print(f"🎛️ 전역 랜덤성 강도 설정: {cls.GLOBAL_RANDOMNESS_INTENSITY}")


    @classmethod
    def get_randomness_intensity(cls):
        """현재 랜덤성 강도 반환"""
        return cls.GLOBAL_RANDOMNESS_INTENSITY


    @staticmethod
    def get_enhanced_randomness_config(progress=1.0, intensity_multiplier=1.0):
        """
        통합 랜덤성 설정 반환 - 전역 강도 적용
        
        Args:
            progress: 훈련 진행도 (0.0 ~ 1.0)
            intensity_multiplier: 지역적 배수 (함수별 추가 조정용)
        
        Returns:
            dict: 모든 랜덤성 파라미터가 포함된 설정
        """
        # ✅ 전역 랜덤성 강도 적용
        global_intensity = RobotPhysicsUtils.GLOBAL_RANDOMNESS_INTENSITY
        
        # 전역 강도가 0이면 모든 랜덤성 비활성화
        if global_intensity == 0.0:
            return RobotPhysicsUtils._get_zero_randomness_config()
        
        # 기본 노이즈 스케일 (진행도에 따라 감소)
        base_noise = 1.0 - 0.5 * progress  # 1.0 → 0.5
        final_intensity = base_noise * intensity_multiplier * global_intensity
        
        return {
            # 위치 랜덤성
            'position': {
                'base_noise': 0.15 * final_intensity,
                'extreme_prob': 0.3 * global_intensity,
                'extreme_range': (0.3 * global_intensity, 0.8 * global_intensity)
            },
            
            # 높이 랜덤성
            'height': {
                'base_noise': 0.12 * final_intensity,
                'extreme_prob': 0.25 * global_intensity,
                'extreme_values': [0.15, 0.18, 0.45, 0.50, 0.8, 0.9]  # 절대값이므로 그대로 유지
            },
            
            # 자세 랜덤성 (각도)
            'orientation': {
                'base_noise': 0.5 * final_intensity,
                'extreme_prob': 0.3 * global_intensity,
                'extreme_range': (-0.8 * global_intensity, 0.8 * global_intensity),
                'flip_prob': 0.03 * global_intensity
            },
            
            # 관절 랜덤성
            'joints': {
                'base_noise': 1.5 * final_intensity,
                'extreme_prob': 0.4 * global_intensity,
                'extreme_multiplier': (2.0 * global_intensity, 5.0 * global_intensity),
                'pattern_prob': 0.6 * global_intensity
            },
            
            # 속도 랜덤성
            'velocity': {
                'base_noise': 0.1 * final_intensity,
                'extreme_prob': 0.3 * global_intensity,
                'extreme_range': (1.0 * global_intensity, 4.0 * global_intensity)
            },
            
            # 물리 파라미터 랜덤성
            'physics': {
                'apply_prob': 0.8 * global_intensity,
                'gravity_range': (
                    1.0 - 0.2 * global_intensity,  # 0.8 ~ 1.0
                    1.0 + 0.2 * global_intensity   # 1.0 ~ 1.2
                ),
                'friction_range': (
                    1.0 - 0.4 * global_intensity,  # 0.6 ~ 1.0
                    1.0 + 0.4 * global_intensity   # 1.0 ~ 1.4
                ),
                'mass_range': (
                    1.0 - 0.15 * global_intensity, # 0.85 ~ 1.0
                    1.0 + 0.15 * global_intensity  # 1.0 ~ 1.15
                ),
                'extreme_prob': 0.15 * global_intensity
            }
        }

    

    @staticmethod
    def _get_zero_randomness_config():
        """랜덤성이 완전히 비활성화된 설정"""
        return {
            'position': {
                'base_noise': 0.0,
                'extreme_prob': 0.0,
                'extreme_range': (0.0, 0.0)
            },
            'height': {
                'base_noise': 0.0,
                'extreme_prob': 0.0,
                'extreme_values': []
            },
            'orientation': {
                'base_noise': 0.0,
                'extreme_prob': 0.0,
                'extreme_range': (0.0, 0.0),
                'flip_prob': 0.0
            },
            'joints': {
                'base_noise': 0.0,
                'extreme_prob': 0.0,
                'extreme_multiplier': (0.0, 0.0),
                'pattern_prob': 0.0
            },
            'velocity': {
                'base_noise': 0.0,
                'extreme_prob': 0.0,
                'extreme_range': (0.0, 0.0)
            },
            'physics': {
                'apply_prob': 0.0,
                'gravity_range': (1.0, 1.0),
                'friction_range': (1.0, 1.0),
                'mass_range': (1.0, 1.0),
                'extreme_prob': 0.0
            }
        }



    

    @staticmethod
    def apply_random_position(data, config):
        """랜덤 위치 적용"""
        pos_config = config['position']
        
        # 랜덤성이 0이면 기본 위치 (0, 0) 유지
        if pos_config['base_noise'] == 0.0:
            data.qpos[0] = 0.0
            data.qpos[1] = 0.0
            return
        
        if np.random.random() < pos_config['extreme_prob']:
            # 극단적인 위치
            extreme_range = pos_config['extreme_range']
            if extreme_range[1] > 0:  # 범위가 유효할 때만
                extreme_pos = np.random.uniform(*extreme_range)
                direction = np.random.choice([-1, 1])
                axis = np.random.choice([0, 1])
                data.qpos[axis] = extreme_pos * direction
                data.qpos[1-axis] = np.random.uniform(-pos_config['base_noise'], pos_config['base_noise'])
            else:
                data.qpos[0] = 0.0
                data.qpos[1] = 0.0
        else:
            # 일반적인 위치
            data.qpos[0] = np.random.uniform(-pos_config['base_noise'], pos_config['base_noise'])
            data.qpos[1] = np.random.uniform(-pos_config['base_noise'], pos_config['base_noise'])

    @staticmethod
    def apply_random_height(data, base_height, config):
        """랜덤 높이 적용"""
        height_config = config['height']
        
        # 랜덤성이 0이면 기본 높이 유지
        if height_config['base_noise'] == 0.0:
            data.qpos[2] = base_height
            return
        
        if np.random.random() < height_config['extreme_prob'] and height_config['extreme_values']:
            # 극단적인 높이
            data.qpos[2] = np.random.choice(height_config['extreme_values'])
        else:
            # 일반적인 높이 변동
            height_noise = np.random.uniform(-height_config['base_noise'], height_config['base_noise'])
            data.qpos[2] = base_height + height_noise

    @staticmethod
    def apply_random_orientation(data, base_pitch=0.0, config=None):
        """랜덤 자세 적용"""
        orient_config = config['orientation']
        
        # 랜덤성이 0이면 기본 자세 유지
        if orient_config['base_noise'] == 0.0:
            r = Rotation.from_euler('xyz', [0.0, base_pitch, 0.0])
            quat = r.as_quat()
            data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
            quat_norm = np.linalg.norm(data.qpos[3:7])
            data.qpos[3:7] /= quat_norm
            return
        
        # 기본 각도 노이즈
        if np.random.random() < orient_config['extreme_prob']:
            # 극단적인 각도
            pitch_noise = np.random.uniform(*orient_config['extreme_range'])
            roll_noise = np.random.uniform(*orient_config['extreme_range'])
            yaw_noise = np.random.uniform(-1.0 * RobotPhysicsUtils.GLOBAL_RANDOMNESS_INTENSITY, 
                                        1.0 * RobotPhysicsUtils.GLOBAL_RANDOMNESS_INTENSITY)
        else:
            # 일반적인 각도 노이즈
            noise_range = orient_config['base_noise']
            pitch_noise = np.random.uniform(-noise_range, noise_range)
            roll_noise = np.random.uniform(-noise_range, noise_range)
            yaw_noise = np.random.uniform(-noise_range, noise_range)
        
        pitch_angle = base_pitch + pitch_noise
        
        # 매우 드물게 완전히 뒤집힌 상태
        if np.random.random() < orient_config['flip_prob']:
            pitch_angle += np.random.choice([np.pi, -np.pi])
        
        # 쿼터니언 변환
        r = Rotation.from_euler('xyz', [roll_noise, pitch_angle, yaw_noise])
        quat = r.as_quat()
        data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        
        # 정규화
        quat_norm = np.linalg.norm(data.qpos[3:7])
        data.qpos[3:7] /= quat_norm

    @staticmethod
    def apply_random_joints(data, base_joints, joint_ranges, config):
        """랜덤 관절 각도 적용"""
        joint_config = config['joints']
        
        # 랜덤성이 0이면 기본 관절 각도 유지
        if joint_config['base_noise'] == 0.0:
            data.qpos[7:19] = base_joints
            return
        
        joint_noise = np.zeros(12)
        
        # 기본 노이즈 적용
        for i in range(12):
            base_range = joint_config['base_noise']
            range_multiplier = np.random.uniform(0.5, 2.0)
            
            if np.random.random() < joint_config['extreme_prob']:
                # 극단적인 노이즈
                extreme_mult = np.random.uniform(*joint_config['extreme_multiplier'])
                if extreme_mult > 0:  # 유효한 배수일 때만
                    joint_noise[i] = np.random.uniform(-base_range * extreme_mult, base_range * extreme_mult)
            else:
                # 일반적인 노이즈
                joint_noise[i] = np.random.uniform(-base_range * range_multiplier, base_range * range_multiplier)
        
        # 랜덤 패턴 적용
        if np.random.random() < joint_config['pattern_prob']:
            RobotPhysicsUtils._apply_joint_patterns(joint_noise, joint_config)
        
        # 최종 관절 각도 설정
        joint_targets = base_joints + joint_noise
        joint_targets = np.clip(joint_targets, 
                               joint_ranges[:, 0] * 0.95, 
                               joint_ranges[:, 1] * 0.95)
        data.qpos[7:19] = joint_targets

    @staticmethod
    def _apply_joint_patterns(joint_noise, config):
        """관절 패턴 적용"""
        patterns = ['symmetric', 'asymmetric', 'diagonal', 'crossed', 'extreme_selection']
        pattern = np.random.choice(patterns)
        
        noise_scale = config['base_noise']
        
        if pattern == 'symmetric':
            sym_noise = np.random.uniform(-noise_scale, noise_scale)
            joint_noise[0:3] += sym_noise   # FR
            joint_noise[3:6] += sym_noise   # FL
            joint_noise[6:9] += sym_noise   # RR
            joint_noise[9:12] += sym_noise  # RL
            
        elif pattern == 'asymmetric':
            left_noise = np.random.uniform(-noise_scale * 2, noise_scale * 2)
            right_noise = np.random.uniform(-noise_scale * 2, noise_scale * 2)
            joint_noise[0:3] += left_noise    # FR
            joint_noise[6:9] += left_noise    # RR
            joint_noise[3:6] += right_noise   # FL
            joint_noise[9:12] += right_noise  # RL
            
        elif pattern == 'diagonal':
            diag1 = np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5)
            diag2 = np.random.uniform(-noise_scale * 1.5, noise_scale * 1.5)
            joint_noise[0:3] += diag1   # FR
            joint_noise[9:12] += diag1  # RL
            joint_noise[3:6] += diag2   # FL
            joint_noise[6:9] += diag2   # RR
            
        elif pattern == 'crossed':
            front_noise = np.random.uniform(-noise_scale * 2, noise_scale * 2)
            rear_noise = -front_noise * np.random.uniform(0.5, 1.5)
            joint_noise[0:6] += front_noise   # 앞다리
            joint_noise[6:12] += rear_noise   # 뒷다리

    @staticmethod
    def apply_random_velocity(data, config):
        """랜덤 속도 적용"""
        vel_config = config['velocity']
        base_noise = vel_config['base_noise']
        
        # 기본 속도 노이즈
        for i in range(len(data.qvel)):
            vel_multiplier = np.random.uniform(0.5, 2.0)
            data.qvel[i] = np.random.normal(0, base_noise * vel_multiplier)
        
        # 극단적인 운동 추가
        if np.random.random() < vel_config['extreme_prob']:
            motion_types = ['spin', 'fall', 'jump', 'slide']
            motion = np.random.choice(motion_types)
            extreme_range = vel_config['extreme_range']
            
            if motion == 'spin':
                axis = np.random.choice([3, 4, 5])
                data.qvel[axis] = np.random.uniform(-extreme_range[1], extreme_range[1])
            elif motion == 'fall':
                data.qvel[1] = np.random.uniform(-extreme_range[0], extreme_range[0])
            elif motion == 'jump':
                data.qvel[2] = np.random.uniform(extreme_range[0], extreme_range[1])
            elif motion == 'slide':
                axis = np.random.choice([0, 1])
                data.qvel[axis] = np.random.uniform(-extreme_range[1], extreme_range[1])

    @staticmethod
    def apply_physics_randomization(model, original_gravity, config):
        """물리 파라미터 랜덤화"""
        phys_config = config['physics']
        
        if np.random.random() < phys_config['apply_prob']:
            # 중력 변화
            if np.random.random() < phys_config['extreme_prob']:
                gravity_scale = np.random.choice([0.3, 0.5, 1.8, 2.5])
            else:
                gravity_scale = np.random.uniform(*phys_config['gravity_range'])
            model.opt.gravity[:] = original_gravity * gravity_scale
            
            # 마찰 변화
            if np.random.random() < phys_config['extreme_prob']:
                friction_scale = np.random.choice([0.1, 0.3, 2.0, 3.0])
            else:
                friction_scale = np.random.uniform(*phys_config['friction_range'])
            
            for i in range(model.ngeom):
                if hasattr(model, 'geom_friction'):
                    model.geom_friction[i, :] *= friction_scale
            
            # 질량 변화
            if np.random.random() < phys_config['extreme_prob']:
                mass_scale = np.random.uniform(0.5, 2.0)
            else:
                mass_scale = np.random.uniform(*phys_config['mass_range'])
            
            for i in range(model.nbody):
                if model.body_mass[i] > 0:
                    model.body_mass[i] *= mass_scale
    
    @staticmethod
    def quat_to_rotmat(quat):
        """Quaternion을 rotation matrix로 변환"""
        w, x, y, z = quat
        return np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
        ])
    
    @staticmethod
    def get_foot_contacts(model, data):
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
    
    @staticmethod
    def get_com_position(model, data):
        """무게중심 위치"""
        return data.xpos[1]  # root body의 위치
    
    @staticmethod
    def get_front_feet_heights(model, data):
        """앞발들의 높이 계산"""
        front_feet_heights = []
        for foot_name in ["FR", "FL"]:
            try:
                foot_site_id = model.site(foot_name).id
                front_feet_heights.append(data.site_xpos[foot_site_id][2])
            except KeyError:
                front_feet_heights.append(0.0)
        return front_feet_heights
    
    @staticmethod
    def get_rear_feet_positions(model, data):
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
    
    @staticmethod
    def get_front_feet_horizontal_velocities(model, data):
        """앞발들의 수평 속도 계산"""
        h_vels = []
        # geom 기반으로 속도를 얻기 위해 mj_objectVelocity 사용
        for foot_name in ["FR", "FL"]:
            try:
                geom_id = model.geom(foot_name).id
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, vel, 0)
                h_vels.append(np.linalg.norm(vel[:2])) # x,y 선속도
            except KeyError:
                h_vels.append(0.0)
        return np.array(h_vels)
    
    @staticmethod
    def get_rear_feet_contact(model, data):
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

    def compute_reward(self, model, data, action):
        """4족 보행 보상 계산"""
        total_reward = 0.0
        reward_info = {}

        # --- 1. 주요 물리량 사전 계산 ---
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2] # 로봇의 Z축 벡터 (up-vector)
        
        # Pitch 각도 계산 (4족 보행에서 중요)
        pitch_angle = np.arcsin(-trunk_rotation_matrix[0, 2])
        
        trunk_height = data.qpos[2]
        com_position = RobotPhysicsUtils.get_com_position(model, data)
        
        front_feet_heights = RobotPhysicsUtils.get_front_feet_heights(model, data)
        rear_feet_positions = RobotPhysicsUtils.get_rear_feet_positions(model, data)

        # --- 2. 핵심 보상 (Positive Rewards) ---

        # [보상 1] 상체 직립 (Torso Upright) - 4족 보행용
        upright_reward = up_vector[2]
        total_reward += self.weights['upright'] * upright_reward
        reward_info['reward_upright'] = upright_reward * self.weights['upright']

        # [보상 2] 목표 높이 유지 (Height) - 4족 보행 높이
        target_height = 0.30  # 4족 보행 목표 높이
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-15 * height_error)
        total_reward += self.weights['height'] * height_reward
        reward_info['reward_height'] = height_reward * self.weights['height']

        # [보상 3] 무게중심 안정성 (CoM over Support Polygon)
        support_center = np.mean(rear_feet_positions, axis=0)
        com_xy = com_position[:2]
        com_error = np.linalg.norm(com_xy - support_center)
        com_reward = np.exp(-15 * com_error)
        total_reward += self.weights['balance'] * com_reward
        reward_info['reward_com_support'] = com_reward * self.weights['balance']
        
        # [보상 4] 발 접촉 (Foot Contact) - 4족 보행에서는 모든 발이 접촉
        foot_contacts = RobotPhysicsUtils.get_foot_contacts(model, data)
        contact_reward = np.mean(foot_contacts)
        total_reward += self.weights['foot_contact'] * contact_reward
        reward_info['reward_foot_contact'] = contact_reward * self.weights['foot_contact']

        # --- 3. 페널티 (Negative Rewards) ---
        # [페널티 1] 과도한 상체 회전 속도 (Angular Velocity)
        angular_vel_penalty = np.sum(np.square(data.qvel[3:6]))
        total_reward += self.weights['energy'] * angular_vel_penalty
        reward_info['penalty_angular_vel'] = self.weights['energy'] * angular_vel_penalty
        
        # [페널티 2] 불필요한 수평 이동 (Horizontal Velocity)
        horizontal_vel_penalty = np.sum(np.square(data.qvel[:2]))
        total_reward += self.weights['energy'] * horizontal_vel_penalty
        reward_info['penalty_horizontal_vel'] = self.weights['energy'] * horizontal_vel_penalty
        
        # [페널티 3] 관절 한계 (Joint Limit)
        joint_pos = data.qpos[7:]
        joint_ranges = model.jnt_range[1:]
        limit_penalty = 0.0
        for i, pos in enumerate(joint_pos):
            if pos < joint_ranges[i, 0] * 0.95:
                limit_penalty += (joint_ranges[i, 0] - pos)**2
            elif pos > joint_ranges[i, 1] * 0.95:
                limit_penalty += (pos - joint_ranges[i, 1])**2
        total_reward += self.weights['joint_limit'] * limit_penalty
        reward_info['penalty_joint_limit'] = self.weights['joint_limit'] * limit_penalty

        total_reward = max(0, total_reward)

        return total_reward, reward_info


class BipedWalkingReward:
    """2족 보행을 위한 보상 함수"""
    
    def __init__(self):
        self.weights = {
            # --- 핵심 성공 요소 (Key Success Factors) ---
            'torso_upright': 15.0,
            'height': 8.0,
            
            # ✅ [수정] 새로운 보상 가중치 추가
            'com_alignment': 12.0,       # 무게중심을 뒷다리 쪽으로 정렬
            'stability': 7.0,            # 안정적으로 균형 유지
            
            'front_feet_up': 10.0,
            'survival_bonus': 5.0,

            # --- 안정화 및 페널티 (Stabilization & Penalties) ---
            'angular_vel_penalty': -0.05,
            'horizontal_vel_penalty': -0.04,
            'action_rate_penalty': -0.01,
            'energy_penalty': -0.005,
            'joint_limit_penalty': -2.0,
            'foot_scuff_penalty': -3.0
        }
        # 이전 액션 저장을 위한 변수
        self._last_action = None


    def compute_reward(self, model, data, action):
        """2족 보행 보상 계산 (개선된 버전)"""
        total_reward = 0.0
        reward_info = {}

        # --- 1. 주요 물리량 사전 계산 ---
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        trunk_height = data.qpos[2]
        com_position = RobotPhysicsUtils.get_com_position(model, data)
        
        front_feet_heights = RobotPhysicsUtils.get_front_feet_heights(model, data)
        rear_feet_positions = RobotPhysicsUtils.get_rear_feet_positions(model, data)
        
        angular_velocity = np.linalg.norm(data.qvel[3:6])

        # --- 2. 핵심 보상 (Positive Rewards) ---

        # [보상] 생존 보너스
        survival_reward = self.weights['survival_bonus']
        total_reward += survival_reward
        reward_info['reward_survival'] = survival_reward

        # [보상] 상체 직립
        upright_reward = up_vector[2]
        total_reward += self.weights['torso_upright'] * upright_reward
        reward_info['reward_upright'] = upright_reward * self.weights['torso_upright']

        # [보상] 목표 높이 유지
        target_height = 0.48
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-15 * height_error)
        total_reward += self.weights['height'] * height_reward
        reward_info['reward_height'] = height_reward * self.weights['height']

        # ✅ [추가 보상 1] 무게중심 정렬 (일어서기 준비 동작)
        support_center_x = np.mean([pos[0] for pos in rear_feet_positions])
        com_x = com_position[0]
        alignment_error = abs(com_x - support_center_x)
        com_alignment_reward = np.exp(-25 * alignment_error)
        total_reward += self.weights['com_alignment'] * com_alignment_reward
        reward_info['reward_com_alignment'] = com_alignment_reward * self.weights['com_alignment']
        
        # ✅ [추가 보상 2] 안정성 (낮은 각속도 유지)
        stability_reward = np.exp(-2.0 * angular_velocity)
        total_reward += self.weights['stability'] * stability_reward
        reward_info['reward_stability'] = stability_reward * self.weights['stability']
        
        # [보상] 앞발 들기
        avg_front_feet_height = np.mean(front_feet_heights)
        front_feet_reward = np.tanh(avg_front_feet_height / 0.15)
        total_reward += self.weights['front_feet_up'] * front_feet_reward
        reward_info['reward_front_feet_up'] = front_feet_reward * self.weights['front_feet_up']

        # --- 3. 페널티 (Negative Rewards) ---
        # [페널티] 과도한 상체 회전 속도
        angular_vel_penalty = np.sum(np.square(data.qvel[3:6]))
        total_reward += self.weights['angular_vel_penalty'] * angular_vel_penalty
        reward_info['penalty_angular_vel'] = self.weights['angular_vel_penalty'] * angular_vel_penalty
        
        # [페널티] 불필요한 수평 이동
        horizontal_vel_penalty = np.sum(np.square(data.qvel[:2]))
        total_reward += self.weights['horizontal_vel_penalty'] * horizontal_vel_penalty
        reward_info['penalty_horizontal_vel'] = self.weights['horizontal_vel_penalty'] * horizontal_vel_penalty
        
        # [페널티] 액션 변화율
        if self._last_action is not None:
            action_rate_penalty = np.sum(np.square(action - self._last_action))
            total_reward += self.weights['action_rate_penalty'] * action_rate_penalty
            reward_info['penalty_action_rate'] = self.weights['action_rate_penalty'] * action_rate_penalty
        self._last_action = action

        # [페널티] 에너지
        energy_penalty = np.sum(np.square(data.ctrl))
        total_reward += self.weights['energy_penalty'] * energy_penalty
        reward_info['penalty_energy'] = self.weights['energy_penalty'] * energy_penalty

        # [페널티] 관절 한계
        joint_pos = data.qpos[7:]
        joint_ranges = model.jnt_range[1:]
        limit_penalty = 0.0
        for i, pos in enumerate(joint_pos):
            if pos < joint_ranges[i, 0] * 0.95:
                limit_penalty += (joint_ranges[i, 0] - pos)**2
            elif pos > joint_ranges[i, 1] * 0.95:
                limit_penalty += (pos - joint_ranges[i, 1])**2
        total_reward += self.weights['joint_limit_penalty'] * limit_penalty
        reward_info['penalty_joint_limit'] = self.weights['joint_limit_penalty'] * limit_penalty
        
        # [페널티] 앞발 쓸림
        front_feet_h_vel = RobotPhysicsUtils.get_front_feet_horizontal_velocities(model, data)
        scuff_penalty = np.sum(front_feet_h_vel * (np.array(front_feet_heights) < 0.05))
        total_reward += self.weights['foot_scuff_penalty'] * scuff_penalty
        reward_info['penalty_foot_scuff'] = self.weights['foot_scuff_penalty'] * scuff_penalty

        total_reward = max(0, total_reward)

        return total_reward, reward_info


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
            foot_contacts = np.array(RobotPhysicsUtils.get_foot_contacts(self.model, self.data))
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

    def _is_initial_pose_unstable(self):
        """초기 자세가 너무 불안정한지 확인"""
        # 무게중심이 지지 다각형을 너무 벗어난 경우
        com_position = RobotPhysicsUtils.get_com_position(self.model, self.data)
        rear_feet_positions = RobotPhysicsUtils.get_rear_feet_positions(self.model, self.data)
        support_center = np.mean(rear_feet_positions, axis=0)
        com_error = np.linalg.norm(com_position[:2] - support_center)
        
        return com_error > 0.15  # 15cm 이상 벗어나면 불안정

    def _set_bipedal_ready_pose_conservative(self):
        """보수적인 2족 준비 자세 (fallback)"""
        # 기본값에 가까운 안전한 설정
        self.data.qpos[0:2] = 0.0  # x, y
        self.data.qpos[2] = 0.62   # z
        
        # 안정적인 pitch 각도
        pitch_angle = -1.5
        r = Rotation.from_euler('xyz', [0, pitch_angle, 0])
        quat = r.as_quat()
        self.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        
        # 공통 기본 관절 각도 사용
        self.data.qpos[7:19] = RobotPhysicsUtils.BIPEDAL_READY_JOINTS.copy()
        
        # 속도는 모두 0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)

    def _set_bipedal_ready_pose(self):
        """2족 보행 준비 자세 설정 - 통합 랜덤성 적용"""
        
        # 훈련 진행도 계산
        progress = min(getattr(self, 'total_timesteps', 0) / self.max_training_timesteps, 1.0)
        
        # ✅ 파라미터명 수정: local_multiplier -> intensity_multiplier
        config = RobotPhysicsUtils.get_enhanced_randomness_config(progress, intensity_multiplier=2.5)
        
        # 위치 랜덤화
        RobotPhysicsUtils.apply_random_position(self.data, config)
        
        # 높이 랜덤화 (2족용)
        RobotPhysicsUtils.apply_random_height(self.data, base_height=0.62, config=config)
        
        # 자세 랜덤화 (2족용 pitch)
        RobotPhysicsUtils.apply_random_orientation(self.data, base_pitch=-1.5, config=config)
        
        # 관절 랜덤화 (2족용)
        base_joints = RobotPhysicsUtils.BIPEDAL_READY_JOINTS.copy()
        joint_ranges = self.model.jnt_range[1:]
        RobotPhysicsUtils.apply_random_joints(self.data, base_joints, joint_ranges, config)
        
        # 속도 랜덤화
        RobotPhysicsUtils.apply_random_velocity(self.data, config)
        
        # 초기화
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)
        
        # 30% 확률로만 안정성 체크 (전역 강도에 따라 조정)
        if np.random.random() < 0.5 * RobotPhysicsUtils.GLOBAL_RANDOMNESS_INTENSITY and self._is_initial_pose_unstable():
            self._set_bipedal_ready_pose_conservative()

    def get_pose_info(self):
        """현재 자세 정보 반환 (디버깅용)"""
        trunk_quat = self.data.qpos[3:7]
        pitch, roll, yaw = self._quat_to_euler(trunk_quat)
        
        return {
            'height': self.data.qpos[2],
            'pitch_degrees': np.rad2deg(pitch),
            'roll_degrees': np.rad2deg(roll),
            'yaw_degrees': np.rad2deg(yaw),
            'pose_type': self._classify_pose(pitch)
        }

    def _classify_pose(self, pitch_rad):
        """Pitch 각도에 따른 자세 분류"""
        pitch_deg = np.rad2deg(pitch_rad)
        
        if -10 <= pitch_deg <= 10:
            return "4족 서기 (수평)"
        elif -100 <= pitch_deg <= -80:
            return "2족 서기 (수직)"
        elif -80 <= pitch_deg <= -10:
            return "중간 자세 (기울어짐)"
        else:
            return "비정상 자세"

    def _set_natural_standing_pose(self):
        """자연스러운 4족 서있기 자세 설정 - 통합 랜덤성 적용"""
        
        # 훈련 진행도 계산
        progress = min(getattr(self, 'total_timesteps', 0) / self.max_training_timesteps, 1.0)
        
        # ✅ 파라미터명 수정: local_multiplier -> intensity_multiplier
        config = RobotPhysicsUtils.get_enhanced_randomness_config(progress, intensity_multiplier=2.0)
        
        # 위치 랜덤화
        RobotPhysicsUtils.apply_random_position(self.data, config)
        
        # 높이 랜덤화
        RobotPhysicsUtils.apply_random_height(self.data, base_height=0.30, config=config)
        
        # 자세 랜덤화
        RobotPhysicsUtils.apply_random_orientation(self.data, base_pitch=0.0, config=config)
        
        # 관절 랜덤화
        base_joints = RobotPhysicsUtils.NATURAL_STANDING_JOINTS.copy()
        joint_ranges = self.model.jnt_range[1:]
        RobotPhysicsUtils.apply_random_joints(self.data, base_joints, joint_ranges, config)
        
        # 속도 랜덤화
        RobotPhysicsUtils.apply_random_velocity(self.data, config)
        
        # 초기화
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)
        
        # 50% 확률로 높이 자동 조정 (전역 강도에 따라 조정)
        if np.random.random() < 0.5 * RobotPhysicsUtils.GLOBAL_RANDOMNESS_INTENSITY:
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
        """물리 파라미터 랜덤화 - 통합 버전"""
        if self.original_gravity is not None:
            progress = min(getattr(self, 'total_timesteps', 0) / self.max_training_timesteps, 1.0)
            # ✅ 파라미터명 수정: local_multiplier -> intensity_multiplier
            config = RobotPhysicsUtils.get_enhanced_randomness_config(progress, intensity_multiplier=1.5)
            
            RobotPhysicsUtils.apply_physics_randomization(self.model, self.original_gravity, config)

    def step(self, action):
        """환경 스텝 실행 - 훈련 진행도 추적 추가"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.standing_reward.compute_reward(self.model, self.data, action)

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
        """종료 조건"""
        
        # 1. 높이 체크
        if self.data.qpos[2] < self._healthy_z_range[0] or self.data.qpos[2] > self._healthy_z_range[1]:
            return True
        
        # 2. 기울기 체크
        trunk_quat = self.data.qpos[3:7]
        pitch, roll, _ = self._quat_to_euler(trunk_quat)
        
        if not (self._healthy_pitch_range[0] <= pitch <= self._healthy_pitch_range[1]):
            return True
            
        if not (self._healthy_roll_range[0] <= roll <= self._healthy_roll_range[1]):
            return True
        
        # 3. 속도 체크
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        
        if linear_vel > 2.0 or angular_vel > 5.0:
            return True
        
        return False

    def _is_standing_successful(self):
        """4족 서있기 성공 판정"""
        trunk_height = self.data.qpos[2]
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]

        # 발 접촉 확인
        foot_contacts = RobotPhysicsUtils.get_foot_contacts(self.model, self.data)

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
        return RobotPhysicsUtils.get_foot_contacts(self.model, self.data)

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

        reward, reward_info = self.bipedal_reward.compute_reward(self.model, self.data, action)

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

    def _is_terminated(self):
        """2족 보행용 종료 조건 (2족 자세에 맞게 수정)"""
        
        # 1. 높이 체크 - 2족 보행 허용 범위 (기존: 0.35 ~ 0.90)
        if self.data.qpos[2] < 0.30 or self.data.qpos[2] > 0.95: # ✅ [수정] 허용 높이 범위 확장
            #print(f"훈련 종료! 높이 초과: {self.data.qpos[2]}")
            return True
        
        # 2. Pitch 각도 체크 - 2족 보행 허용 범위
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        pitch_angle = np.arcsin(-trunk_rotation_matrix[2, 0])

        # 목표 pitch: -1.5 라디안 (약 -86도)
        # (기존: -1.7 ~ -1.3 라디안)
        if pitch_angle < -1.9 or pitch_angle > -1.1: # ✅ [수정] Pitch 허용 각도 범위 확장
            #print(f"훈련 종료! Pitch 각도 초과: {pitch_angle}")
            return True
        
        # 3. Roll 각도 체크 - 좌우 기울기 
        roll_angle = np.arctan2(trunk_rotation_matrix[2, 1], trunk_rotation_matrix[2, 2])
        if abs(roll_angle) > np.deg2rad(35):  # ✅ [수정] Roll 허용 각도를 45도로 확장
            #print(f"훈련 종료! Roll 각도 초과: {roll_angle}")
            return True
        
        # 4. 속도 체크 (기존: linear_vel > 2.0)
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        
        if linear_vel > 2.5 or angular_vel > 6.0: # ✅ [수정] 선속도 및 각속도 허용치 증가
            #print(f"훈련 종료! 속도 초과: {linear_vel}, {angular_vel}")
            return True
        
        # 5. 안정성 체크 (수정 없음)
        if not hasattr(self, '_instability_count'):
            self._instability_count = 0
            
        if self._is_unstable():
            self._instability_count += 1
            if self._instability_count > 50:
                #print(f"훈련 종료! 안정성 초과: {self._instability_count}")
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

    def _is_bipedal_success(self):
        """2족 보행 성공 판정 - 실제 2족 자세 기준으로 수정"""
        
        # 1. 높이 확인
        trunk_height = self.data.qpos[2]
        height_ok = 0.58 < trunk_height < 0.68  # 2족 보행 높이
        
        # 2. Pitch 각도 확인
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        pitch_angle = np.arcsin(-trunk_rotation_matrix[2, 0]) # ✅ 올바른 수식으로 수정
        pitch_ok = -1.6 < pitch_angle < -1.4  # 목표 주변 ±0.1 라디안
        
        # 3. 앞발이 충분히 들려있는지
        front_feet_heights = [
            self._get_foot_height('FR'),
            self._get_foot_height('FL')
        ]
        front_feet_up = all(h > 0.15 for h in front_feet_heights)  # 15cm 이상
        
        # 4. 뒷발만 접촉
        rear_contacts = [
            self._is_foot_contact('RR'),
            self._is_foot_contact('RL')
        ]
        front_contacts = [
            self._is_foot_contact('FR'),
            self._is_foot_contact('FL')
        ]
        rear_feet_only = all(rear_contacts) and not any(front_contacts)
        
        # 5. 안정성
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        stable = angular_vel < 1.5 and linear_vel < 0.5
        
        # 6. 지속 시간
        duration_ok = self.episode_length > 300  # 3초 이상
        
        return (height_ok and pitch_ok and front_feet_up and 
                rear_feet_only and stable and duration_ok)


class BipedalCurriculumEnv(BipedalWalkingEnv):
    """2족 보행 커리큘럼 환경"""

    def __init__(self, curriculum_stage=0, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_bipedal_curriculum()

    def _setup_bipedal_curriculum(self):
        """2족 보행 단계별 커리큘럼 - 점진적 난이도 증가"""
        
        if self.curriculum_stage == 0:
            # Stage 0: 무게중심 이동 학습 (4족에서 시작)
            self.target_height = 0.40
            self.target_pitch = -0.5  # 약 -29도
            self.front_feet_target = 0.03
            self.stability_threshold = 5.0
            
        elif self.curriculum_stage == 1:
            # Stage 1: 중간 자세
            self.target_height = 0.48
            self.target_pitch = -0.8  # 약 -46도
            self.front_feet_target = 0.08
            self.stability_threshold = 4.0
            
        elif self.curriculum_stage == 2:
            # Stage 2: 반직립 자세
            self.target_height = 0.55
            self.target_pitch = -1.2  # 약 -69도
            self.front_feet_target = 0.12
            self.stability_threshold = 3.0
            
        else:
            # Stage 3+: 완전한 2족 자세
            self.target_height = 0.62
            self.target_pitch = -1.5  # 약 -86도
            self.front_feet_target = 0.18
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
                
        except Exception as e:
            print(f"⚠️ 모델 분석 실패: {e}, 기본 설정 사용")
            env_kwargs['use_base_observation'] = False
    
    return env_class(**env_kwargs)