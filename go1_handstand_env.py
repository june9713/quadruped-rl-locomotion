#!/usr/bin/env python3
"""
Go1 물구나무서기 환경
"""

import numpy as np
import mujoco
from go1_mujoco_env import Go1MujocoEnv

class HandstandReward:
    """물구나무서기를 위한 보상 함수"""
    
    def __init__(self):
        # 보상 가중치들
        self.weights = {
            'orientation': 10.0,     # 몸체 방향 보상
            'height': 5.0,          # 높이 보상
            'stability': 3.0,       # 안정성 보상
            'feet_contact': 8.0,    # 발 접촉 보상 (물구나무서기에서는 penalty)
            'head_contact': 5.0,    # 머리/몸체 접촉 보상
            'balance': 4.0,         # 균형 보상
            'energy': -0.1          # 에너지 효율성
        }
    
    def compute_reward(self, model, data):
        """물구나무서기 보상 계산"""
        total_reward = 0.0
        
        # 1. 몸체 방향 보상 (뒤집혀야 함)
        trunk_quat = data.qpos[3:7]  # 몸체 quaternion
        trunk_rotation_matrix = self._quat_to_rotmat(trunk_quat)
        
        # z축이 아래쪽을 향해야 함 (물구나무서기)
        up_vector = trunk_rotation_matrix[:, 2]  # z축 방향
        desired_up = np.array([0, 0, -1])  # 아래쪽
        
        orientation_reward = np.dot(up_vector, desired_up)
        orientation_reward = max(0, orientation_reward)  # 음수 방지
        total_reward += self.weights['orientation'] * orientation_reward
        
        # 2. 높이 보상 (몸체가 적절한 높이에 있어야 함)
        trunk_height = data.qpos[2]  # z 좌표
        target_height = 0.4  # 물구나무서기 시 적절한 몸체 높이
        height_diff = abs(trunk_height - target_height)
        height_reward = np.exp(-5 * height_diff)  # 목표 높이에 가까울수록 높은 보상
        total_reward += self.weights['height'] * height_reward
        
        # 3. 발 접촉 패널티 (발이 땅에 닿으면 안됨)
        foot_contacts = self._get_foot_contacts(model, data)
        foot_penalty = sum(foot_contacts)  # 발이 땅에 닿을수록 페널티
        total_reward -= self.weights['feet_contact'] * foot_penalty
        
        # 4. 머리/몸체 접촉 보상 (머리나 몸체가 땅에 닿아야 함)
        head_contact = self._get_head_contact(model, data)
        total_reward += self.weights['head_contact'] * head_contact
        
        # 5. 균형 보상 (너무 빠르게 움직이지 않아야 함)
        trunk_vel = data.qvel[:3]  # 선형 속도
        trunk_angular_vel = data.qvel[3:6]  # 각속도
        
        velocity_magnitude = np.linalg.norm(trunk_vel)
        angular_velocity_magnitude = np.linalg.norm(trunk_angular_vel)
        
        stability_reward = np.exp(-2 * velocity_magnitude) * np.exp(-2 * angular_velocity_magnitude)
        total_reward += self.weights['stability'] * stability_reward
        
        # 6. 에너지 효율성 (너무 많은 힘을 사용하지 않도록)
        motor_efforts = np.square(data.ctrl).sum()
        total_reward += self.weights['energy'] * motor_efforts
        
        return total_reward
    
    def _quat_to_rotmat(self, quat):
        """Quaternion을 rotation matrix로 변환"""
        w, x, y, z = quat
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])
    
    def _get_foot_contacts(self, model, data):
        """발 접촉 감지"""
        foot_names = ["FR", "FL", "RR", "RL"]  # Go1 발 이름들
        contacts = []
        
        for foot_name in foot_names:
            try:
                foot_geom_id = model.geom(foot_name).id
                # 접촉 감지
                contact = False
                for i in range(data.ncon):
                    contact_geom1 = data.contact[i].geom1
                    contact_geom2 = data.contact[i].geom2
                    
                    if contact_geom1 == foot_geom_id or contact_geom2 == foot_geom_id:
                        contact = True
                        break
                
                contacts.append(1.0 if contact else 0.0)
            except:
                contacts.append(0.0)
        
        return contacts
    
    def _get_head_contact(self, model, data):
        """머리/몸체 접촉 감지"""
        try:
            # 몸체 접촉 확인
            for i in range(data.ncon):
                contact_geom1 = data.contact[i].geom1
                contact_geom2 = data.contact[i].geom2
                
                # 바닥과의 접촉인지 확인 (geom id 0이 보통 바닥)
                if contact_geom1 == 0 or contact_geom2 == 0:
                    # 발이 아닌 다른 부위의 접촉인지 확인
                    geom_id = contact_geom1 if contact_geom1 != 0 else contact_geom2
                    geom_name = model.geom(geom_id).name
                    
                    # 발이 아닌 몸체 부분의 접촉이면 보상
                    if geom_name and not any(foot in geom_name for foot in ["FR", "FL", "RR", "RL"]):
                        return 1.0
            
            return 0.0
        except:
            return 0.0

class Go1HandstandEnv(Go1MujocoEnv):
    """물구나무서기를 학습하는 Go1 환경"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handstand_reward = HandstandReward()
        self.episode_length = 0
        self.max_episode_length = 1000
        
    def reset(self, **kwargs):
        """환경 리셋 - 물구나무서기 시작 자세로"""
        obs, info = super().reset(**kwargs)
        
        # 물구나무서기 시작 자세 설정
        self._set_handstand_initial_pose()
        
        self.episode_length = 0
        return self._get_obs(), info
    
    def _set_handstand_initial_pose(self):
        """물구나무서기 초기 자세 설정"""
        # 몸체를 뒤집어서 시작
        self.data.qpos[0:3] = [0, 0, 0.4]  # x, y, z 위치
        
        # 180도 회전 (물구나무서기)
        # Quaternion for 180 degree rotation around x-axis: [0, 1, 0, 0]
        self.data.qpos[3:7] = [0, 1, 0, 0]  # w, x, y, z quaternion
        
        # 다리를 위로 접어서 균형 잡기 좋은 자세
        joint_angles = [
            0.0, -1.5, 2.5,   # FR leg (hip, thigh, calf)
            0.0, -1.5, 2.5,   # FL leg
            0.0, -1.5, 2.5,   # RR leg  
            0.0, -1.5, 2.5,   # RL leg
        ]
        
        # 관절 각도 설정
        self.data.qpos[7:7+len(joint_angles)] = joint_angles
        
        # 속도 초기화
        self.data.qvel[:] = 0
        
        # 물리 시뮬레이션 한 스텝 실행
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, action):
        """환경 스텝 실행"""
        # 액션 실행
        self.do_simulation(action, self.frame_skip)
        
        # 관찰값 계산
        obs = self._get_obs()
        
        # 보상 계산
        reward = self.handstand_reward.compute_reward(self.model, self.data)
        
        # 종료 조건 확인
        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length
        
        self.episode_length += 1
        
        info = {
            'episode_length': self.episode_length,
            'handstand_reward': reward,
            'handstand_success': self._is_handstand_successful()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _is_terminated(self):
        """종료 조건 - 로봇이 넘어지거나 너무 낮아지면 종료"""
        trunk_height = self.data.qpos[2]
        
        # 너무 낮아지면 종료
        if trunk_height < 0.1:
            return True
            
        # 몸체가 너무 기울어지면 종료
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.handstand_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        # z축이 거의 수평이 되면 넘어진 것으로 간주
        if up_vector[2] > -0.3:  # 거의 수평
            return True
            
        return False
    
    def _is_handstand_successful(self):
        """물구나무서기 성공 판정"""
        trunk_height = self.data.qpos[2]
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.handstand_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        # 조건: 적절한 높이 + 뒤집힌 상태 + 오래 버티기
        height_ok = 0.3 < trunk_height < 0.6
        orientation_ok = up_vector[2] < -0.7  # 충분히 뒤집힘
        duration_ok = self.episode_length > 100  # 100 스텝 이상 버팀
        
        return height_ok and orientation_ok and duration_ok

# 점진적 보상을 지원하는 환경
class GradualHandstandEnv(Go1HandstandEnv):
    """점진적으로 보상을 바꾸는 환경"""
    
    def __init__(self, reward_mix={"original": 0.5, "handstand": 0.5}, **kwargs):
        super().__init__(**kwargs)
        self.reward_mix = reward_mix
    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        # 원래 보상 계산 (간단한 서있기 보상)
        original_reward = self._compute_original_reward()
        
        # 물구나무서기 보상 계산
        handstand_reward = self.handstand_reward.compute_reward(self.model, self.data)
        
        # 보상 혼합
        mixed_reward = (
            self.reward_mix["original"] * original_reward +
            self.reward_mix["handstand"] * handstand_reward
        )
        
        info.update({
            "original_reward": original_reward,
            "handstand_reward": handstand_reward,
            "mixed_reward": mixed_reward
        })
        
        return obs, mixed_reward, terminated, truncated, info
    
    def _compute_original_reward(self):
        """원래 환경의 보상 계산 (걷기, 서있기 등)"""
        # 간단한 서있기 + 안정성 보상
        trunk_height = self.data.qpos[2]
        height_reward = 1.0 if trunk_height > 0.3 else 0.0
        
        # 속도 보상 (원래는 이동 보상이었음)
        trunk_vel = self.data.qvel[:2]  # x, y 속도
        velocity_penalty = -0.1 * np.sum(np.square(trunk_vel))
        
        return height_reward + velocity_penalty