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
__all__ = ['Go1StandingEnv', 'GradualStandingEnv', 'StandingReward']


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

    def _set_natural_standing_pose(self):
        """✅ 자연스러운 4족 서있기 자세 설정"""
        
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
        """종료 조건 - 물구나무서기 방지 추가"""
        # 기본 건강 상태 확인
        if not self.is_healthy:
            return True

        # ✅ 물구나무서기 즉시 종료
        if self.data.qpos[2] < 0.15:  # 높이가 15cm 이하
            print("🚨 물구나무서기 감지 - 에피소드 종료")
            return True

        # 높이 체크 (너무 낮거나 높으면 종료)
        if self.data.qpos[2] < 0.25:
            return True

        # 뒤집힌 상태 체크
        trunk_quat = self.data.qpos[3:7]
        trunk_rotation_matrix = self.standing_reward._quat_to_rotmat(trunk_quat)
        up_vector = trunk_rotation_matrix[:, 2]
        
        if up_vector[2] < 0.7:  # 너무 기울어짐
            return True

        # 너무 빠른 움직임 체크
        linear_vel = np.linalg.norm(self.data.qvel[:3])
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        
        if linear_vel > 2.0 or angular_vel > 5.0:
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