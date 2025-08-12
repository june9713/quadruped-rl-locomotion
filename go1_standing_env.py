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
import traceback

# visual_train.py에서 import할 수 있도록 환경 이름 추가
__all__ = [
           'BipedWalkingReward', 'BipedalWalkingEnv', 'BipedalCurriculumEnv',
           'create_compatible_env']


class RobotPhysicsUtils:
    """로봇 물리 계산을 위한 공통 유틸리티 클래스"""

    GLOBAL_RANDOMNESS_INTENSITY = 0.0  # 기본값 1.0 (0.0 = 랜덤성 없음, 2.0 = 2배 강화)

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
        0.0, 2.0, -0.6,    # FR
        0.0, 2.0, -0.6,    # FL
        # 뒷다리 (RR, RL) - 몸을 지지하기 좋게 굽힌 상태  
        0.0, 2.5, -1.0,     # RR
        0.0, 2.5, -1.0,     # RL
    ])

    @staticmethod
    def set_random_joint_angles(data, model):
        """
        (사용자 요청) 매 스텝마다 각 관절의 '각도(위치)'를 관절 범위 내의
        완전히 새로운 랜덤 값으로 '설정'합니다.
        """
        try:
            # 12개 관절의 위치(qpos) 인덱스는 7부터 18까지입니다.
            # 해당 관절의 범위(jnt_range)는 model.jnt_range[1:]에 해당합니다 (root joint 제외).
            joint_ranges = model.jnt_range[1:]
            
            # 각 관절의 유효 범위 내에서 독립적인 랜덤 각도를 생성합니다.
            random_angles = np.random.uniform(low=joint_ranges[:, 0], high=joint_ranges[:, 1])
            
            # 계산된 랜덤 각도를 관절 위치(qpos)에 직접 덮어씁니다.
            data.qpos[7:19] = random_angles
        except Exception as e:
            # 함수가 실패하더라도 시뮬레이션이 중단되지 않도록 방지
            pass


    @staticmethod
    def apply_step_joint_velocity_noise(data, total_timesteps, max_training_timesteps):
        """
        (대폭 수정) 매 스텝마다 로봇 관절에 '매우 강하고 예측 불가능한' 속도 노이즈를 가하여,
        극단적인 상황에 대한 대처 능력을 학습시킵니다.
        """
        intensity = RobotPhysicsUtils.get_randomness_intensity()
        if intensity <= 0.0:
            return

        # ✅ [수정] 확률적 적용을 제거하고 '매 스텝마다' 노이즈를 적용합니다.
        # 훈련 진행도에 따른 감소는 유지하되, 노이즈의 기본 크기를 대폭 상향합니다.
        progress = min(1.0, total_timesteps / max_training_timesteps)
        
        try:
            # ✅ [수정] 노이즈 기본 크기를 0.75 -> 2.5로 대폭 상향하여 격렬한 움직임을 만듭니다.
            # 훈련 초반(progress=0)에 매우 강한 노이즈를 인가하고, 훈련이 진행되면 점차 줄여나갑니다.
            max_noise_magnitude = 2.5 * intensity * (1 - progress**2) 
            
            # 12개 관절에 대한 랜덤 속도 노이즈 생성
            joint_vel_noise = np.random.uniform(-max_noise_magnitude, max_noise_magnitude, 12)

            # 기존 관절 속도에 노이즈를 더해 강제로 움직임을 망가뜨립니다.
            data.qvel[6:] += joint_vel_noise
            
        except Exception as e:
            # 함수가 실패하더라도 시뮬레이션이 중단되지 않도록 방지
            print(traceback.format_exc())

    @staticmethod
    def get_rear_feet_velocities(model, data):
        """뒷발들의 월드 좌표계 기준 선속도(xyz)를 반환"""
        velocities = []
        # geom 기반으로 속도를 얻기 위해 mj_objectVelocity 사용
        for foot_name in ["RR", "RL"]:
            try:
                geom_id = model.geom(foot_name).id
                vel = np.zeros(6)
                # com_based=0: 월드 좌표계 기준 속도
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, vel, 0)
                velocities.append(vel[:3]) # 선속도 (vx, vy, vz)
            except KeyError:
                velocities.append(np.zeros(3))
        return velocities

    @staticmethod
    def get_rear_leg_part_positions(model, data):
        """뒷다리의 고관절, 무릎, 발의 월드 좌표(xyz)를 반환"""
        part_positions = {}
        leg_parts = {
            'hip': ["RR_hip", "RL_hip"],    # 고관절 body 이름
            'knee': ["RR_calf", "RL_calf"], # 무릎 body 이름 (calf body가 무릎 관절 위치)
            'foot': ["RR", "RL"]            # 발 site 이름
        }
        
        # 고관절 (hip) 위치
        hip_pos = []
        for name in leg_parts['hip']:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                hip_pos.append(data.xpos[body_id])
            except:
                hip_pos.append(np.zeros(3))
        part_positions['hip'] = hip_pos

        # 무릎 (knee) 위치
        knee_pos = []
        for name in leg_parts['knee']:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                knee_pos.append(data.xpos[body_id])
            except:
                knee_pos.append(np.zeros(3))
        part_positions['knee'] = knee_pos

        # 발 (foot) 위치
        foot_pos = []
        for name in leg_parts['foot']:
            try:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                foot_pos.append(data.site_xpos[site_id])
            except:
                foot_pos.append(np.zeros(3))
        part_positions['foot'] = foot_pos

        return part_positions


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
        통합 랜덤성 설정 반환 - 전역 강도 적용 (수정된 버전)
        
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
        
        # ⚠️ [수정] 위치/높이 랜덤성 대폭 축소. 평평한 지형에서는 큰 의미가 없기 때문입니다.
        #    대신 자세, 관절, 물리 랜덤성에 집중하여 강인한 정책을 학습합니다.
        position_intensity = 0.1 # 기존 1.0 -> 0.1 (90% 감소)
        height_intensity = 0.2   # 기존 1.0 -> 0.2 (80% 감소)
        
        return {
            # 위치 랜덤성 (매우 약하게 설정)
            'position': {
                'base_noise': 0.05 * position_intensity,
                'extreme_prob': 0.1 * position_intensity,
                'extreme_range': (0.1 * position_intensity, 0.2 * position_intensity)
            },
            
            # 높이 랜덤성 (매우 약하게 설정)
            'height': {
                'base_noise': 0.05 * height_intensity,
                'extreme_prob': 0.1 * height_intensity,
                'extreme_values': [0.28, 0.32, 0.58, 0.65] 
            },
            
            # 자세 랜덤성 (각도) - 중요하므로 유지 및 강화
            'orientation': {
                'base_noise': 0.6 * final_intensity,
                'extreme_prob': 0.3 * global_intensity,
                'extreme_range': (-0.9 * global_intensity, 0.9 * global_intensity),
                'flip_prob': 0.02 * global_intensity # 뒤집힐 확률은 낮춤
            },
            
            # 관절 랜덤성 - 중요하므로 유지 및 강화
            'joints': {
                'base_noise': 1.8 * final_intensity,
                'extreme_prob': 0.4 * global_intensity,
                'extreme_multiplier': (2.5 * global_intensity, 6.0 * global_intensity),
                'pattern_prob': 0.6 * global_intensity
            },
            
            # 속도 랜덤성
            'velocity': {
                'base_noise': 0.15 * final_intensity,
                'extreme_prob': 0.3 * global_intensity,
                'extreme_range': (1.5 * global_intensity, 4.5 * global_intensity)
            },
            
            # 물리 파라미터 랜덤성 - 중요하므로 유지
            'physics': {
                'apply_prob': 0.8 * global_intensity,
                'gravity_range': (
                    1.0 - 0.2 * global_intensity,
                    1.0 + 0.2 * global_intensity
                ),
                'friction_range': (
                    1.0 - 0.4 * global_intensity,
                    1.0 + 0.4 * global_intensity
                ),
                'mass_range': (
                    1.0 - 0.15 * global_intensity,
                    1.0 + 0.15 * global_intensity
                ),
                'extreme_prob': 0.15 * global_intensity
            }
        }



    @staticmethod
    def apply_adaptive_step_noise(data, model, total_timesteps, max_training_timesteps):
        """
        (수정) 훈련 진행도에 따라 노이즈 종류와 강도를 점진적으로 변경합니다.
        - 초기: '물리 기반 토크 충격'으로 관절을 흔들어 강한 탐험 유도
        - 후기: 물리 기반 속도 노이즈 및 외력 (안정화 및 세밀한 제어 학습)
        """
        # =========================================================================
        # ✅ [사용자 요청] 랜덤 확률 기반으로 노이즈 적용 여부 결정
        # 아래 apply_prob 값을 조절하여 노이즈가 적용될 확률을 테스트할 수 있습니다.
        # (예: 1.0 = 매번 적용, 0.7 = 70% 확률로 적용, 0.1 = 10% 확률로 적용)
        # =========================================================================
        apply_prob = 0.5  # <--- 이 값을 수정하여 확률을 직접 테스트하세요.
        if np.random.random() > apply_prob:
            return # 설정된 확률에 따라 노이즈를 적용하지 않고 건너뜁니다.


        intensity = RobotPhysicsUtils.get_randomness_intensity()
        if intensity <= 0.0:
            return

        # 1. 훈련 진행도 계산 (0.0에서 1.0으로 증가)
        progress = min(1.0, total_timesteps / max_training_timesteps)

        # 2. 노이즈 가중치 계산
        # 초반에 강하고 빠르게 감소하는 '초기 탐험용' 노이즈 가중치
        initial_exploration_weight = (1.0 - progress)**3
        # 서서히 강해지는 '물리 기반' 노이즈 가중치
        physical_noise_weight = progress

        # --- [핵심 수정] '관절 위치 강제 변경' 대신 '관절 토크(힘) 적용' 방식 (훈련 초반 집중) ---
        if initial_exploration_weight > 0.01: # 가중치가 거의 0이면 연산 생략
            try:
                # 1. 관절에 가할 '충격'의 기본 크기를 설정합니다.
                #    이 값은 로봇의 PD 제어기를 이겨내고 움직임을 만들어낼 만큼 충분히 커야 합니다.
                #    이 값은 튜닝이 필요한 하이퍼파라미터입니다.
                force_magnitude = 50.0 * initial_exploration_weight * intensity
                
                # 2. 12개 관절에 대해 [-force_magnitude, force_magnitude] 범위의 독립적인 랜덤 토크를 생성합니다.
                joint_force_shock = np.random.uniform(-force_magnitude, force_magnitude, 12)
                
                # 3. 계산된 랜덤 토크를 data.qfrc_applied에 더해줍니다.
                #    qfrc_applied는 MuJoCo가 매 스텝 계산하는 제어 토크에 추가적인 외력을 더하는 역할을 합니다.
                #    이는 물리적으로 타당한 방식으로 관절에 '충격'을 주는 것과 같습니다.
                data.qfrc_applied[0:12] += joint_force_shock

            except Exception:
                # 함수가 실패하더라도 시뮬레이션이 중단되지 않도록 방지
                pass

        # --- 물리 기반 노이즈 (훈련 후반에 집중) ---
        if physical_noise_weight > 0.01: # 가중치가 거의 0이면 연산 생략
            # 가. 속도 노이즈
            try:
                # 기존 속도 노이즈 크기에 physical_noise_weight를 곱해 강도를 점진적으로 높입니다.
                max_vel_noise = 2.5 * intensity * physical_noise_weight
                joint_vel_noise = np.random.uniform(-max_vel_noise, max_vel_noise, 12)
                data.qvel[6:] += joint_vel_noise
            except Exception:
                pass
            
            # 나. 몸통 외력 (기존 코드 유지)
            try:
                pass
                # 외력이 가해질 확률과 크기 역시 physical_noise_weight에 비례하여 점진적으로 높입니다.
                #perturb_prob = 0.05 * intensity * physical_noise_weight
                #if np.random.random() < perturb_prob:
                    #trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                    #if trunk_id != -1:
                        #max_force = 75.0 * intensity * physical_noise_weight
                        #force = np.random.uniform(-max_force, max_force, 3)
                        #force[2] *= 0.2
                        #data.xfrc_applied[trunk_id, :3] += force
            except Exception:
                pass

    
    @staticmethod
    def apply_step_perturbations(model, data):
        """
        매 스텝마다 로봇 몸통에 랜덤한 외력을 가하여 동적 안정성 학습을 강화합니다.
        (새로 추가된 함수)
        """
        intensity = RobotPhysicsUtils.get_randomness_intensity()
        if intensity <= 0.0:
            return

        # 외력을 가할 확률 (너무 자주 가하면 학습이 불안정해질 수 있음)
        # 강도 1.0 기준, 5% 확률로 외력 적용
        perturb_prob = 0.05 * intensity 
        if np.random.random() > perturb_prob:
            return
            
        try:
            trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            if trunk_id == -1:
                return

            # 외력의 최대 크기 (강도에 비례)
            max_force = 75.0 * intensity  # 2족 보행 시 강하게 밀리도록 상향 조정
            
            # 랜덤한 방향으로 힘 생성
            #force = np.random.uniform(-max_force, max_force, 3)
            #force[2] *= 0.2 # 수직 방향 힘은 약하게 적용 (주로 수평으로 밀도록)

            # 기존 외력에 추가 (덮어쓰지 않음)
            #data.xfrc_applied[trunk_id, :3] += force
            
        except Exception as e:
            # 함수가 실패하더라도 시뮬레이션이 중단되지 않도록 방지
            pass


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
    def apply_step_joint_position_noise(data, total_timesteps, max_training_timesteps):
        """
        (신규 추가) 매 스텝마다 로봇 관절 '위치'에 직접 노이즈를 가하여,
        수동적인 자세를 적극적으로 방해하고 강인한 복원력을 학습시킵니다.
        """
        intensity = RobotPhysicsUtils.get_randomness_intensity()
        if intensity <= 0.0:
            return

        progress = min(1.0, total_timesteps / max_training_timesteps)
        
        try:
            # 위치(각도)에 대한 노이즈이므로 속도 노이즈보다 훨씬 작은 값을 사용해야 합니다.
            # 0.05 rad는 약 2.8도에 해당하며, intensity와 곱해져 효과를 냅니다.
            max_noise_magnitude = 0.05 * intensity * (1 - progress**2)
            
            joint_pos_noise = np.random.uniform(-max_noise_magnitude, max_noise_magnitude, 12)

            # 기존 관절 위치(qpos)에 직접 노이즈를 더해 자세를 강제로 계속 바꿉니다.
            # 이는 물리적으로는 부정확하지만, 에이전트가 특정 자세에 안주하는 것을 방지하는 강력한 수단입니다.
            data.qpos[7:19] += joint_pos_noise
            
        except Exception as e:
            # 함수가 실패하더라도 시뮬레이션이 중단되지 않도록 방지
            pass
    
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



class BipedWalkingReward:
    """
    2족 보행을 위한 보상 함수 (동적 안정성 강화 버전)
    - 비현실적인 '제자리 유지'를 '무게중심 안정성' 보상으로 대체합니다.
    """
    
    def __init__(self):
        self.weights = {
            # --- 1. 자세 유지 보상 (안정적인 서기) ---
            'survival_bonus': 0.5,
            'torso_upright': 3.0,
            'height': 2.5,
            'front_feet_up': 2.0,
            'leg_extension': 1.5,
            'swing_speed_reward': 1.5,
            'leg_posture_hierarchy': 1.5,  # ✅ [추가] 다리 자세 계층 구조 보상 가중치

            # --- 2. 동적 안정성 및 걷기 보상 ---
            'forward_velocity': 3.0,
            'stepping': 4.0,
            'com_stability': 2.5,
            'angular_velocity_reward': 2.0,

            # --- 3. 페널티 ---
            'action_rate_penalty': -0.002,
            'energy_penalty': -0.005,
            'joint_limit_penalty': -2.0,
            'foot_scuff_penalty': -1.5,
            'low_height_penalty': -10.0,
            'rear_calf_contact_penalty': -5.0,
        }
        
        self._last_action = None
        self.target_forward_velocity = 0.0
        self.rear_feet_air_time = np.zeros(2)
        self.calf_geom_ids = None

    def compute_reward(self, model, data, action, dt, total_timesteps):
        total_reward = 0.0
        reward_info = {}

        if self.calf_geom_ids is None:
            calf_geom_names = ["RR_calf_geom1", "RR_calf_geom2", "RL_calf_geom1", "RL_calf_geom2"]
            self.calf_geom_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in calf_geom_names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) != -1}

        # --- 주요 물리량 사전 계산 ---
        trunk_quat = data.qpos[3:7]
        trunk_rotation_matrix = RobotPhysicsUtils.quat_to_rotmat(trunk_quat)
        trunk_height = data.qpos[2]
        rear_feet_contact = np.array(RobotPhysicsUtils.get_rear_feet_contact(model, data))
        is_contact = rear_feet_contact > 0.1

        ##############################################################
        ### --- 1단계: 안정적인 자세 유지 (핵심 보상) --- ###
        ##############################################################
        total_reward += self.weights['survival_bonus']
        reward_info['reward_survival'] = self.weights['survival_bonus']
        
        target_pitch = -1.5
        current_pitch = np.arcsin(-trunk_rotation_matrix[2, 0])
        pitch_error = abs(current_pitch - target_pitch)
        upright_reward = np.exp(-3.0 * pitch_error) * self.weights['torso_upright']
        total_reward += upright_reward
        reward_info['reward_upright'] = upright_reward
        
        target_height = 0.62
        height_error = abs(trunk_height - target_height)
        height_reward = np.exp(-10.0 * height_error) * self.weights['height']
        total_reward += height_reward
        reward_info['reward_height'] = height_reward

        front_feet_heights = RobotPhysicsUtils.get_front_feet_heights(model, data)
        avg_front_feet_height = np.mean(front_feet_heights)
        front_feet_reward = np.tanh(avg_front_feet_height / 0.15) * self.weights['front_feet_up']
        total_reward += front_feet_reward
        reward_info['reward_front_feet_up'] = front_feet_reward

        leg_pos = RobotPhysicsUtils.get_rear_leg_part_positions(model, data)
        hip_knee_dist_rr = np.linalg.norm(leg_pos['hip'][0] - leg_pos['knee'][0])
        hip_knee_dist_rl = np.linalg.norm(leg_pos['hip'][1] - leg_pos['knee'][1])
        avg_leg_extension = (hip_knee_dist_rr + hip_knee_dist_rl) / 2
        leg_extension_reward = avg_leg_extension * self.weights['leg_extension']
        total_reward += leg_extension_reward
        reward_info['reward_leg_extension'] = leg_extension_reward

        # ✅ --- [추가] 고관절 > 무릎 > 발 높이 순서 보상 ---
        leg_posture_reward = 0.0
        # leg_pos는 위에서 이미 계산됨
        for i in range(2):  # 0: Right-Rear, 1: Left-Rear
            hip_z = leg_pos['hip'][i][2]
            knee_z = leg_pos['knee'][i][2]
            foot_z = leg_pos['foot'][i][2]

            # 고관절이 무릎보다 높은지 확인 (차이가 클수록 좋음)
            leg_posture_reward += np.tanh(max(0, hip_z - knee_z)) * 0.25
            
            # 무릎이 발보다 높은지 확인 (차이가 클수록 좋음)
            leg_posture_reward += np.tanh(max(0, knee_z - foot_z)) * 0.25
        
        # leg_posture_reward는 최대 0.5 (tanh의 최대값은 1)
        final_leg_posture_reward = leg_posture_reward * self.weights['leg_posture_hierarchy']
        total_reward += final_leg_posture_reward
        reward_info['reward_leg_posture'] = final_leg_posture_reward
        
        #####################################################################
        ### --- 2단계: 동적 안정성 및 걷기 학습 --- ###
        #####################################################################
        rear_feet_pos = RobotPhysicsUtils.get_rear_feet_positions(model, data)
        support_center = np.mean(rear_feet_pos, axis=0)
        com_xy = data.qpos[:2]
        com_error = np.linalg.norm(com_xy - support_center)
        com_stability_reward = np.exp(-10.0 * com_error) * self.weights['com_stability']
        total_reward += com_stability_reward
        reward_info['reward_com_stability'] = com_stability_reward
        
        angular_vel = np.linalg.norm(data.qvel[3:5])
        angular_velocity_reward = np.tanh(angular_vel) * self.weights['angular_velocity_reward']
        total_reward += angular_velocity_reward
        reward_info['reward_angular_velocity'] = angular_velocity_reward
        
        first_contact = (self.rear_feet_air_time > 0.0) & is_contact
        self.rear_feet_air_time += dt
        stride_time = np.clip(self.rear_feet_air_time, 0.1, 0.4)
        stepping_reward = np.sum(stride_time * first_contact) * self.weights['stepping']
        self.rear_feet_air_time[is_contact] = 0.0
        total_reward += stepping_reward
        reward_info['reward_stepping'] = stepping_reward
        
        is_airborne = ~np.array(is_contact, dtype=bool) 
        swing_speed_reward = 0.0
        if np.any(is_airborne):
            rear_feet_vels = RobotPhysicsUtils.get_rear_feet_velocities(model, data)
            airborne_feet_vels = np.array(rear_feet_vels)[is_airborne]
            horizontal_speeds = np.linalg.norm(airborne_feet_vels[:, :2], axis=1)
            avg_swing_speed = np.mean(horizontal_speeds)
            swing_speed_reward = np.tanh(avg_swing_speed) * self.weights.get('swing_speed_reward', 0.0)
        total_reward += swing_speed_reward
        reward_info['reward_swing_speed'] = swing_speed_reward
        
        #####################################################################
        ### --- 3단계: 커리큘럼 및 페널티 --- ###
        #####################################################################
        if total_timesteps > 1_000_000:
            self.target_forward_velocity = 0.5
        
        if self.target_forward_velocity > 0:
            forward_vel_error = abs(data.qvel[0] - self.target_forward_velocity)
            forward_reward = np.exp(-5.0 * forward_vel_error) * self.weights['forward_velocity']
            total_reward += forward_reward
            reward_info['reward_forward_velocity'] = forward_reward
        
        low_height_penalty = min(0, trunk_height - 0.35) * self.weights['low_height_penalty']
        total_reward += low_height_penalty
        reward_info['penalty_low_height'] = low_height_penalty
        
        ground_geom_id = 0
        calf_contact_count = sum(1 for i in range(data.ncon) if (data.contact[i].geom1 in self.calf_geom_ids and data.contact[i].geom2 == ground_geom_id) or (data.contact[i].geom2 in self.calf_geom_ids and data.contact[i].geom1 == ground_geom_id))
        rear_calf_penalty = calf_contact_count * self.weights['rear_calf_contact_penalty']
        total_reward += rear_calf_penalty
        reward_info['penalty_rear_calf_contact'] = rear_calf_penalty
        
        energy_penalty = np.sum(np.square(data.ctrl)) * self.weights['energy_penalty']
        total_reward += energy_penalty
        reward_info['penalty_energy'] = energy_penalty

        if self._last_action is not None:
            action_rate_penalty = np.sum(np.square(action - self._last_action)) * self.weights['action_rate_penalty']
            total_reward += action_rate_penalty
            reward_info['penalty_action_rate'] = action_rate_penalty
        self._last_action = action
        
        return total_reward, reward_info



class BipedalWalkingEnv(Go1MujocoEnv):
    """
    2족 보행 전용 환경 (Go1StandingEnv 의존성 제거)
    Go1MujocoEnv를 직접 상속받아 독립적으로 작동합니다.
    """

    def __init__(self, **kwargs):
        # ------------------------------------------------------------------
        # region: Go1StandingEnv의 __init__ 로직 병합
        # ------------------------------------------------------------------
        
        # ✅ [수정] 멀티프로세싱 환경에서도 랜덤 강도가 올바르게 설정되도록 수정
        # 환경 생성 시 전달된 'randomness_intensity' 값을 가져와 설정합니다.
        # 이 코드를 통해 각 자식 프로세스가 자신의 랜덤 강도를 명확히 인지하게 됩니다.
        randomness_intensity = kwargs.get('randomness_intensity', 1.5)
        RobotPhysicsUtils.set_randomness_intensity(randomness_intensity)
        
        filtered_kwargs = {}
        allowed_params = {
            'randomize_physics', 'render_mode', 'frame_skip',
            'observation_space', 'default_camera_config'
        }
        for key, value in kwargs.items():
            if key in allowed_params:
                filtered_kwargs[key] = value

        self._use_base_observation = kwargs.get('use_base_observation', False)
        
        # 부모 클래스를 Go1MujocoEnv로 직접 지정하여 초기화
        super().__init__(**filtered_kwargs)
        
        # 2족 보행용 보상 함수 사용
        self.bipedal_reward = BipedWalkingReward()
        self.episode_length = 0
        self.max_episode_length = 1000

        # Domain randomization 설정
        self.randomize_physics = kwargs.get('randomize_physics', True)
        self.original_gravity = None

        # 훈련 진행도 추적
        self.total_timesteps = 0
        self.max_training_timesteps = 5_000_000

        # 관찰 공간 재설정
        if self._use_base_observation:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self._get_base_obs().shape,
                dtype=np.float64
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self._get_extended_obs().shape,
                dtype=np.float64
            )
        # endregion
        # ------------------------------------------------------------------
        
        # 2족 보행을 위한 건강 상태 조건 (BipedalWalkingEnv 고유 설정)
        self._healthy_z_range = (0.25, 0.70) # 더 넓은 높이 허용
        self._healthy_pitch_range = (-np.deg2rad(140), -np.deg2rad(30))
        self._healthy_roll_range = (-np.deg2rad(170), np.deg2rad(170))
        
        self._last_x_position = 0.0
        self._no_progress_steps = 0
        self.episode_success_rate = 0.0


    # ------------------------------------------------------------------
    # region: Go1StandingEnv로부터 가져온 헬퍼 함수들
    # ------------------------------------------------------------------

    def _get_obs(self):
        """관찰 상태 반환 - 호환성 모드에 따라 선택"""
        if self._use_base_observation:
            return self._get_base_obs()
        else:
            return self._get_extended_obs()

    def _get_base_obs(self):
        """기본 Go1MujocoEnv와 호환되는 관찰 상태 (45차원)"""
        return super()._get_obs()

    def _get_extended_obs(self):
        """확장된 관찰 상태 (2족 보행용 추가 정보 포함)"""
        base_obs = self._get_base_obs()
        foot_heights = np.array([
            self._get_foot_height('FR'), self._get_foot_height('FL'),
            self._get_foot_height('RR'), self._get_foot_height('RL')
        ])
        foot_contacts = np.array(RobotPhysicsUtils.get_foot_contacts(self.model, self.data))
        trunk_quat = self.data.qpos[3:7]
        pitch, roll = self._quat_to_euler(trunk_quat)[:2]
        target_height = 0.62
        height_error = abs(self.data.qpos[2] - target_height)
        
        extended_info = np.concatenate([
            foot_heights, foot_contacts, [pitch, roll], [height_error]
        ])
        return np.concatenate([base_obs, extended_info])

    def _get_foot_height(self, foot_name):
        """발 높이 계산"""
        try:
            return self.data.site_xpos[self.model.site(foot_name).id][2]
        except:
            return 0.0

    def _quat_to_euler(self, quat):
        """Quaternion을 Euler angles로 변환"""
        w, x, y, z = quat
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _apply_domain_randomization(self):
        """물리 파라미터 랜덤화"""
        if self.original_gravity is not None:
            progress = min(getattr(self, 'total_timesteps', 0) / self.max_training_timesteps, 1.0)
            config = RobotPhysicsUtils.get_enhanced_randomness_config(progress, intensity_multiplier=1.5)
            RobotPhysicsUtils.apply_physics_randomization(self.model, self.original_gravity, config)

    def _ensure_rear_feet_contact(self):
        """뒷발이 지면에 접촉하도록 로봇 높이 자동 조정"""
        try:
            foot_positions_z = [self.data.site_xpos[self.model.site(name).id][2] for name in ["RR", "RL"]]
            if foot_positions_z:
                lowest_foot_z = min(foot_positions_z)
                height_adjustment = 0.005 - lowest_foot_z
                self.data.qpos[2] += height_adjustment
                mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            print(f"⚠️ 뒷발 높이 자동 조정 실패: {e}")

    def _is_initial_pose_unstable(self):
        """초기 자세가 너무 불안정한지 확인"""
        com_pos = RobotPhysicsUtils.get_com_position(self.model, self.data)
        rear_feet_pos = RobotPhysicsUtils.get_rear_feet_positions(self.model, self.data)
        support_center = np.mean(rear_feet_pos, axis=0)
        com_error = np.linalg.norm(com_pos[:2] - support_center)
        return com_error > 0.20 # 허용 오차 약간 증가

    def _set_bipedal_ready_pose_conservative(self):
        """보수적인 2족 준비 자세 (안전 장치) - 상속 문제 해결을 위해 BipedalWalkingEnv에서 복사"""
        self.data.qpos[0:2] = 0.0
        self.data.qpos[2] = 0.62
        pitch_angle = -1.5
        r = Rotation.from_euler('xyz', [0, pitch_angle, 0])
        quat = r.as_quat()
        self.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        self.data.qpos[7:19] = RobotPhysicsUtils.BIPEDAL_READY_JOINTS.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_bipedal_ready_pose(self):
        """(수정) 2족 보행 준비 자세 설정 - 초기 자세 랜덤성 제거"""
        # 사용자 요청: 초기 자세의 랜덤성을 제거하여 항상 동일한 자세에서 시작하도록 수정합니다.
        # 위치 및 자세, 관절 각도, 속도를 고정된 값으로 설정합니다.
        self.data.qpos[0:2] = 0.0  # x, y 위치 초기화
        self.data.qpos[2] = 0.62   # z 높이 설정
        
        # 고정된 피치 각도(-1.5 rad, 약 -86도)로 설정
        pitch_angle = -1.5
        r = Rotation.from_euler('xyz', [0, pitch_angle, 0])
        quat = r.as_quat()
        # MuJoCo 쿼터니언 순서 (w, x, y, z)
        self.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        
        # 기본 'BIPEDAL_READY_JOINTS' 관절 각도로 설정
        self.data.qpos[7:19] = RobotPhysicsUtils.BIPEDAL_READY_JOINTS.copy()
        
        # 모든 속도를 0으로 초기화
        self.data.qvel[:] = 0.0
        
        # 가속도 및 제어 입력 초기화
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # 시뮬레이션 상태 업데이트
        mujoco.mj_forward(self.model, self.data)
        
        # 뒷발이 지면에 확실히 닿도록 높이 미세 조정
        #self._ensure_rear_feet_contact()

    def _is_foot_contact(self, foot_name):
        """발 접촉 상태 확인"""
        try:
            foot_geom_id = self.model.geom(foot_name).id
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == foot_geom_id or contact.geom2 == foot_geom_id:
                    # 접촉력이 0.1 이상일 때만 유의미한 접촉으로 간주
                    if np.linalg.norm(mujoco.mj_contactForce(self.model, self.data, i, np.zeros(6))) > 0.1:
                        return True
            return False
        except:
            return False

    # endregion
    # ------------------------------------------------------------------


    def reset(self, seed=None, options=None):
        """환경 리셋 - 2족 보행 준비 자세에서 시작"""
        # Go1MujocoEnv의 reset_model()이 내부적으로 호출됨
        obs, info = super().reset(seed=seed, options=options)

        if self.original_gravity is None:
            self.original_gravity = self.model.opt.gravity.copy()

        self._set_bipedal_ready_pose()

        if self.randomize_physics:
            self._apply_domain_randomization()

        self.episode_length = 0
        self._last_x_position = self.data.qpos[0]
        self._no_progress_steps = 0
        
        # info 딕셔너리에 리셋 정보 추가
        info.update({
            'initial_height': self.data.qpos[2],
            'initial_pitch_deg': np.rad2deg(self._quat_to_euler(self.data.qpos[3:7])[1]),
        })
        
        return self._get_obs(), info

    def step(self, action):
        """환경 스텝 실행 - 훈련 단계에 따라 적응적으로 노이즈 적용"""
        
        self.data.xfrc_applied[:] = 0
        
        # =========================================================================
        # ✅ [사용자 최종 요청 완벽 반영]
        # 훈련 초반에는 관절을 강제로 순간이동시키고,
        # 진행되면서 점차 물리적인 속도/외력 노이즈로 전환하는 '노이즈 커리큘럼' 적용
        # =========================================================================
        if self.randomize_physics:
            RobotPhysicsUtils.apply_adaptive_step_noise(
                self.data,
                self.model,
                getattr(self, 'total_timesteps', 0),
                self.max_training_timesteps
            )

        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        reward, reward_info = self.bipedal_reward.compute_reward(
            self.model, self.data, action, self.dt, getattr(self, 'total_timesteps', 0)
        )

        terminated, reason = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_length
        self.episode_length += 1
        
        if hasattr(self, 'total_timesteps'):
            self.total_timesteps += 1

        if terminated or truncated:
            success = self._is_bipedal_success()
            self.episode_success_rate = 0.95 * self.episode_success_rate + 0.05 * success
            
            if hasattr(self, 'advance_curriculum'):
                self.advance_curriculum(self.episode_success_rate)
                
        info = {
            'episode_length': self.episode_length,
            'bipedal_reward': reward,
            'bipedal_success': self._is_bipedal_success(),
            'termination_reason': reason if terminated else None,
            'current_success_rate': self.episode_success_rate,
            **reward_info
        }

        return obs, reward, terminated, truncated, info

    def _is_terminated(self):
        """2족 보행용 종료 조건"""
        height = self.data.qpos[2]
        if height < self._healthy_z_range[0] or height > self._healthy_z_range[1]:
             return True, f"height_out_of_range ({height:.2f})"

        pitch = self._quat_to_euler(self.data.qpos[3:7])[1]
        if not (self._healthy_pitch_range[0] < pitch < self._healthy_pitch_range[1]):
            return True, f"pitch_out_of_range ({np.rad2deg(pitch):.1f} deg)"

        # [수정] roll 값도 라디안 단위로 가져와서 라디안 범위와 비교
        roll = self._quat_to_euler(self.data.qpos[3:7])[0]
        if not (self._healthy_roll_range[0] < roll < self._healthy_roll_range[1]):
            pass
            # 출력할 때만 각도(degree)로 변환
            #print(f"roll: {np.rad2deg(roll):.1f} deg"  , np.rad2deg(self._healthy_roll_range[0]) , np.rad2deg(self._healthy_roll_range[1]), self._healthy_roll_range[0] < roll  , self._healthy_roll_range[1] > roll)
            #return True, f"roll_out_of_range ({np.rad2deg(roll):.1f} deg)"

        linear_vel = np.linalg.norm(self.data.qvel[:3])
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        if linear_vel > 10.0 or angular_vel > 15.0:
            return True, "excessive_velocity"
            
        # 300스텝(약 3초) 동안 5cm도 전진 못하면 종료
        if self.episode_length > 0 and self.episode_length % 300 == 0:
            if abs(self.data.qpos[0] - self._last_x_position) < 0.05:
                self._no_progress_steps += 1
            else:
                self._no_progress_steps = 0
            self._last_x_position = self.data.qpos[0]

        if self._no_progress_steps >= 1: # 300 스텝 동안 멈춰있으면 종료
            return True, "no_progress"
            
        return False, "not_terminated"

    def _is_bipedal_success(self):
        """2족 보행 성공 판정"""
        height_ok = 0.58 < self.data.qpos[2] < 0.68
        
        pitch = self._quat_to_euler(self.data.qpos[3:7])[1]
        pitch_ok = -1.6 < pitch < -1.4
        
        front_feet_up = all(self._get_foot_height(name) > 0.15 for name in ['FR', 'FL'])
        
        rear_contacts = [self._is_foot_contact('RR'), self._is_foot_contact('RL')]
        front_contacts = [self._is_foot_contact('FR'), self._is_foot_contact('FL')]
        rear_feet_only = all(rear_contacts) and not any(front_contacts)
        
        stable = np.linalg.norm(self.data.qvel[3:6]) < 1.5
        duration_ok = self.episode_length > 300
        
        return all([height_ok, pitch_ok, front_feet_up, rear_feet_only, stable, duration_ok])


class BipedalCurriculumEnv(BipedalWalkingEnv):
    """2족 보행 커리큘럼 환경"""

    def __init__(self, curriculum_stage=0, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_stage = curriculum_stage
        self._setup_bipedal_curriculum()
        self.bipedal_reward = BipedWalkingReward()
        
        # ✅ [개선] 현재 에피소드 성공률과 목표 속도 추적
        self.episode_success_rate = 0.0
        self.bipedal_reward.target_forward_velocity = 0.1 # 초기 목표 속도는 낮게 설정
    
    def advance_curriculum(self):
        """성공률에 따라 목표 속도를 점진적으로 높입니다."""
        if self.episode_success_rate > 0.7 and self.bipedal_reward.target_forward_velocity < 0.6:
            new_vel = self.bipedal_reward.target_forward_velocity + 0.05
            self.bipedal_reward.target_forward_velocity = new_vel
            print(f"🎓 커리큘럼 진행: 목표 속도가 {new_vel:.2f} m/s 로 상향되었습니다.")


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
            self.target_pitch = -2.0  # 약 -69도
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