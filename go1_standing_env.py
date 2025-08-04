#!/usr/bin/env python3
"""
Go1 4족 정상 서있기 환경 - 자연스러운 4족 자세에서 시작
"""

import numpy as np
import mujoco
from go1_mujoco_env import Go1MujocoEnv
import math
from collections import deque

# visual_train.py에서 import할 수 있도록 환경 이름 추가
__all__ = ['Go1StandingEnv', 'GradualStandingEnv', 'StandingReward', 
           'BipedWalkingReward', 'BipedalWalkingEnv', 'BipedalCurriculumEnv']


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
        
        # 7. 에너지 페널티 (2족은 더 많은 토크 허용)
        motor_efforts = np.sum(np.square(data.ctrl))
        energy_penalty = motor_efforts * 0.5  # 페널티 완화
        total_reward += self.weights['energy'] * energy_penalty
        reward_info['energy'] = -energy_penalty
        
        # 8. 단계별 보너스
        stage_bonus = self._compute_stage_bonus(front_feet_height, rear_feet_contact, trunk_height)
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

    def compute_reward(self, model, data):
        """4족 정상 서있기 보상 계산"""
        total_reward = 0.0
        reward_info = {}

        # 1. 똑바로 서있기 보상 (물구나무서기 방지)
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]  # z축 방향

        # ✅ 물구나무서기 감지 및 페널티
        trunk_height = data.qpos[2]
        if trunk_height < 0.15:  # 너무 낮으면 물구나무서기 의심
            upright_reward = -10.0  # 강한 페널티
        elif up_vector[2] > 0.8:  # 정상 방향이고 충분히 직립
            upright_reward = up_vector[2]
        else:
            upright_reward = -0.4
        total_reward += self.weights['upright'] * upright_reward
        reward_info['upright'] = upright_reward

        # 2. 높이 보상 (4족 서있기 기준)
        trunk_height = data.qpos[2]
        target_height = 0.30  # 4족 서있기 목표 높이
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-15 * height_error)
        total_reward += self.weights['height'] * height_reward
        reward_info['height'] = height_reward

        # 3. 발 접촉 보상 (모든 발이 지면에 접촉해야 함)
        foot_contacts = self._get_foot_contacts(model, data)
        # 모든 발이 지면에 닿아야 함
        all_feet_contact = sum(foot_contacts)  # 4개 발 모두
        foot_reward = all_feet_contact / 4.0  # 정규화 (0~1)
        
        total_reward += self.weights['foot_contact'] * foot_reward
        reward_info['foot_contact'] = foot_reward

        # 4. 균형 보상
        trunk_vel = data.qvel[:3]
        trunk_angular_vel = data.qvel[3:6]

        # 너무 빠르게 움직이지 않아야 함 (제자리 서기)
        linear_stability = np.exp(-3 * np.linalg.norm(trunk_vel))  # 모든 방향 제한
        angular_stability = np.exp(-4 * np.linalg.norm(trunk_angular_vel))

        balance_reward = linear_stability * angular_stability
        total_reward += self.weights['balance'] * balance_reward
        reward_info['balance'] = balance_reward

                # 5. 전진 속도 보상 제거 (제자리 서기가 목표)
                # 5. 전진 속도 0 보상 (제자리 서기)
        forward_vel = data.qvel[0]  # x 방향 속도
        forward_vel_reward = np.exp(-10 * forward_vel**2)  # 속도가 0에 가까울수록 높은 보상
        total_reward += 2.0 * forward_vel_reward  # 가중치 2.0
        reward_info['forward_vel'] = forward_vel_reward

        # 6. 좌우 안정성 (옆으로 기울지 않기)
        roll_angle = np.arctan2(up_vector[1], up_vector[2])
        lateral_reward = np.exp(-8 * abs(roll_angle))
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

        # 9. 좌우 대칭성 (4족 보행의 안정성)
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
            if i < len(joint_ranges):
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
        symmetry_reward = np.exp(-1.5 * symmetry_error)

        return symmetry_reward


class Go1StandingEnv(Go1MujocoEnv):
    """4족 정상 서있기 환경 - 자연스러운 4족 자세에서 시작"""

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
            else:
                # 허용되지 않는 파라미터는 무시
                pass
        
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

        #print("🐕 4족 정상 서있기 환경 초기화 완료")

    def _set_bipedal_ready_pose(self):
        """2족 보행 준비 자세 설정"""
        
        #print("🐕 2족 보행 준비 자세로 초기화...")
        
        # 1. 트렁크 위치 - 약간 높게
        self.data.qpos[0] = np.random.uniform(-0.01, 0.01)  # x: 작은 변동
        self.data.qpos[1] = np.random.uniform(-0.01, 0.01)  # y: 작은 변동  
        self.data.qpos[2] = 0.35  # z: 2족 준비 자세 높이 (높게)

        # 2. 트렁크 자세 - 약간 뒤로 기울임
        pitch_angle = np.deg2rad(-5)  # 5도 뒤로
        self.data.qpos[3] = np.cos(pitch_angle/2)  # w
        self.data.qpos[4] = 0.0                    # x
        self.data.qpos[5] = np.sin(pitch_angle/2)  # y  
        self.data.qpos[6] = 0.0                    # z

        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        self.data.qpos[3:7] /= quat_norm

        # 3. 2족 보행 준비 관절 각도
        # Go1 관절 순서: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        #                RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        
        joint_targets = np.array([
            # 앞다리 (FR, FL) - 들기 준비 (약간 굽힘)
            0.0, 0.3, -0.6,    # FR: 들기 준비
            0.0, 0.3, -0.6,    # FL: 좌우 대칭
            
            # 뒷다리 (RR, RL) - 지지 준비 (더 펴짐)
            0.0, 0.4, -0.8,    # RR: 지지 준비
            0.0, 0.4, -0.8     # RL: 좌우 대칭
        ])

        # 작은 노이즈 추가 (자연스러운 변동)
        joint_noise = np.random.normal(0, 0.02, 12)
        joint_targets += joint_noise
        
        # 관절 위치 설정
        self.data.qpos[7:19] = joint_targets

        # 4. 속도 초기화 (정지 상태)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        # 5. 제어 입력 초기화
        self.data.ctrl[:] = 0.0

        # 6. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)

        # 7. 발이 지면에 접촉하도록 높이 자동 조정
        self._auto_adjust_height_for_ground_contact()
        
        #print(f"✅ 2족 보행 준비 자세로 초기화 완료 - 높이: {self.data.qpos[2]:.3f}m")

    def _set_natural_standing_pose(self):
        """✅ 자연스러운 4족 서있기 자세 설정 (기존 메서드 유지)"""
        
        #print("🐕 자연스러운 4족 서있기 자세로 초기화...")
        
        # 1. 트렁크 위치 설정
        self.data.qpos[0] = np.random.uniform(-0.01, 0.01)  # x: 작은 변동
        self.data.qpos[1] = np.random.uniform(-0.01, 0.01)  # y: 작은 변동  
        self.data.qpos[2] = 0.30  # z: 4족 서있는 자세 높이

        # 2. 트렁크 자세 (수평 유지)
        self.data.qpos[3] = 1.0     # w (quaternion)
        self.data.qpos[4] = 0.0     # x 
        self.data.qpos[5] = 0.0     # y
        self.data.qpos[6] = 0.0     # z

        # 쿼터니언 정규화
        quat_norm = np.linalg.norm(self.data.qpos[3:7])
        self.data.qpos[3:7] /= quat_norm

        # 3. 자연스러운 4족 서있기 관절 각도
        # Go1 관절 순서: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        #                RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
        
        joint_targets = np.array([
            # 앞다리 (FR, FL) - 더 안정적으로
            0.0, 0.6, -1.2,    # FR: 덜 굽혀서 안정성 확보
            0.0, 0.6, -1.2,    # FL: 좌우 대칭
            
            # 뒷다리 (RR, RL) - 더 안정적으로
            0.0, 0.8, -1.5,    # RR: 적당히 굽혀서 지지
            0.0, 0.8, -1.5     # RL: 좌우 대칭
        ])

        # 작은 노이즈 추가 (자연스러운 변동)
        joint_noise = np.random.normal(0, 0.02, 12)
        joint_targets += joint_noise
        
        # 관절 위치 설정
        self.data.qpos[7:19] = joint_targets

        # 4. 속도 초기화 (정지 상태)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

        # 5. 제어 입력 초기화
        self.data.ctrl[:] = 0.0

        # 6. 물리 시뮬레이션 적용
        mujoco.mj_forward(self.model, self.data)

        # 7. 발이 지면에 접촉하도록 높이 자동 조정
        self._auto_adjust_height_for_ground_contact()
        
        #print(f"✅ 4족 서있기 자세로 초기화 완료 - 높이: {self.data.qpos[2]:.3f}m")

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
                    print(f"⚠️ {foot_name} 발 위치를 찾을 수 없습니다.")
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
                
                #print(f"  높이 조정: {height_adjustment:.3f}m, 최종 높이: {self.data.qpos[2]:.3f}m")
                
        except Exception as e:
            print(f"⚠️ 높이 자동 조정 실패: {e}")

    def reset(self, seed=None, options=None):
        """환경 리셋 - 자연스러운 4족 서있기 자세에서 시작"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # ✅ 자연스러운 4족 서있기 자세로 설정
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
        """환경 스텝 실행"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_obs()

        reward, reward_info = self.standing_reward.compute_reward(self.model, self.data)

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

    def _is_standing_successful(self):
        """4족 서있기 성공 판정 (기존 메서드 유지)"""
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

    def _get_bipedal_obs(self):
        """2족 보행용 관찰 상태"""
        # 기본 정보
        position = self.data.qpos[7:].flatten()
        velocity = self.data.qvel.flatten()
        
        # 2족 보행 특화 정보 추가
        # 1. 발 높이 정보
        foot_heights = np.array([
            self._get_foot_height('FR'),
            self._get_foot_height('FL'),
            self._get_foot_height('RR'),
            self._get_foot_height('RL')
        ])
        
        # 2. 무게중심 위치
        com_position = self._get_com_position_relative_to_feet()
        
        # 3. 발 접촉 정보
        foot_contacts = self.feet_contact_forces > 0.1
        
        # 4. 상체 기울기
        trunk_quat = self.data.qpos[3:7]
        pitch, roll = self._quat_to_euler(trunk_quat)[:2]
        
        # 5. 목표 자세 (2족 서기)
        target_pose = np.array([0.0, 0.0])  # 목표: 제자리 2족
        
        curr_obs = np.concatenate([
            position,
            velocity[:6] * 0.1,  # 스케일 조정
            velocity[6:],
            foot_heights,
            com_position,
            foot_contacts.astype(float),
            [pitch, roll],
            target_pose,
            self._last_action
        ])
        
        return curr_obs.clip(-self._clip_obs_threshold, self._clip_obs_threshold)

    def _get_foot_height(self, foot_name):
        """발 높이 계산"""
        try:
            foot_site_id = self.model.site(foot_name).id
            foot_pos = self.data.site_xpos[foot_site_id]
            return foot_pos[2]  # z 좌표
        except:
            return 0.0

    def _get_com_position_relative_to_feet(self):
        """무게중심의 발 기준 상대 위치"""
        try:
            # 무게중심 위치
            com_pos = self.data.xpos[1][:2]  # x, y만
            
            # 뒷발 중심점
            rr_pos = self.data.site_xpos[self.model.site("RR").id][:2]
            rl_pos = self.data.site_xpos[self.model.site("RL").id][:2]
            rear_center = (rr_pos + rl_pos) / 2
            
            # 무게중심이 뒷발 중심에서 얼마나 떨어져 있는지
            relative_pos = com_pos - rear_center
            return relative_pos
        except:
            return np.array([0.0, 0.0])

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


class BipedalWalkingEnv(Go1StandingEnv):
    """2족 보행 전용 환경"""

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
            else:
                pass
        
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

        #print("🐕 2족 보행 환경 초기화 완료")

    def reset(self, seed=None, options=None):
        """환경 리셋 - 2족 보행 준비 자세에서 시작"""
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        # ✅ 2족 보행 준비 자세로 설정
        self._set_bipedal_ready_pose()

        if self.randomize_physics and self.original_gravity is not None:
            self._apply_domain_randomization()

        self.episode_length = 0

        return self._get_bipedal_obs(), info

    def step(self, action):
        """환경 스텝 실행"""
        self.do_simulation(action, self.frame_skip)

        obs = self._get_bipedal_obs()

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

    def _is_terminated(self):
        """2족 보행용 종료 조건"""
        
        # 1. 높이 체크 - 범위 확대
        if self.data.qpos[2] < 0.15 or self.data.qpos[2] > 0.6:
            return True
        
        # 2. 기울기 체크 - 더 관대하게
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.bipedal_reward._quat_to_rotmat(trunk_quat)
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
            'observation_space', 'default_camera_config'
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