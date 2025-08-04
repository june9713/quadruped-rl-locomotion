import numpy as np

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
            # 몸체나 머리 부분의 접촉 확인
            trunk_geom_id = model.geom("trunk").id if "trunk" in [model.geom(i).name for i in range(model.ngeom)] else None
            
            if trunk_geom_id is not None:
                for i in range(data.ncon):
                    contact_geom1 = data.contact[i].geom1
                    contact_geom2 = data.contact[i].geom2
                    
                    if contact_geom1 == trunk_geom_id or contact_geom2 == trunk_geom_id:
                        return 1.0
            
            return 0.0
        except:
            return 0.0

# Go1 환경에 물구나무서기 보상 적용
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
            'handstand_reward': reward
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