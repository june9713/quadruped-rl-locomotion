#!/usr/bin/env python3
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from collections import deque
import os
import argparse
import subprocess
import sys
import copy
import imageio.v2 as imageio
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv
import pandas as pd
from visual_training_callback import VisualTrainingCallback, VideoRecordingCallback, EnhancedVisualCallback
import torch
import glob
from collections import deque, defaultdict
try:
    from go1_standing_env import Go1StandingEnv, GradualStandingEnv, BipedalWalkingEnv, RobotPhysicsUtils
except ImportError:
    print("⚠️ go1_standing_env.py 파일이 필요합니다!")
    raise

# 한글 폰트 설정
try:
    plt.rc('font', family='Malgun Gothic')
except:
    print("Malgun Gothic 폰트가 없어 기본 폰트로 설정됩니다.")
plt.rcParams['axes.unicode_minus'] = False


def parse_arguments():
    """명령행 인수 파싱 - 2족 보행 최적화"""
    parser = argparse.ArgumentParser(description='2족 보행 강화학습 시각적 훈련')
    
    parser.add_argument('--task', type=str, default='standing', 
                       help='훈련할 태스크 (기본값: standing)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='사전 훈련된 모델 경로')
    parser.add_argument('--visual_interval', type=float, default=1.0,
                       help='시각화 주기 (분 단위, 기본값: 1.0)')
    parser.add_argument('--show_duration', type=int, default=15,
                       help='시뮬레이션 보여주는 시간 (초 단위, 기본값: 15)')
    parser.add_argument('--save_videos', action='store_true',
                       help='비디오 저장 여부')
    parser.add_argument('--total_timesteps', type=int, default=5_000_000,
                       help='총 훈련 스텝 수 (기본값: 5,000,000)')
    parser.add_argument('--num_envs', type=int, default=12,
                       help='병렬 환경 수 (기본값: 12)')
    parser.add_argument('--video_interval', type=int, default=150_000,
                       help='비디오 녹화 간격 (timesteps, 기본값: 150,000)')
    parser.add_argument('--use_curriculum', action='store_true',
                       help='커리큘럼 학습 사용')
    
    # 2족 보행 최적화된 하이퍼파라미터
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='학습률 (기본값: 2e-4)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='배치 크기 (기본값: 128)')
    parser.add_argument('--n_steps', type=int, default=1024,
                       help='롤아웃 스텝 수 (기본값: 1024)')
    parser.add_argument('--clip_range', type=float, default=0.15,
                       help='PPO 클립 범위 (기본값: 0.15)')
    parser.add_argument('--entropy_coef', type=float, default=0.005,
                       help='엔트로피 계수 (기본값: 0.005)')
    
    # 새로운 2족 보행 특화 파라미터
    parser.add_argument('--target_vel', type=float, default=0.0,
                       help='목표 속도 (기본값: 0.0 - 제자리 서기)')
    parser.add_argument('--stability_weight', type=float, default=1.5,
                       help='안정성 가중치 (기본값: 1.5)')
    parser.add_argument('--height_tolerance', type=float, default=0.12,
                       help='높이 허용 오차 (기본값: 0.12)')
    parser.add_argument('--early_stopping', action='store_true',
                       help='조기 정지 사용 (수렴 시)')
    parser.add_argument('--checkpoint_interval', type=int, default=500_000,
                       help='체크포인트 저장 간격 (기본값: 500,000)')
    
    # 새로운 옵션: 관찰 공간 호환성
    parser.add_argument('--ignore_pretrained_obs_mismatch', action='store_true',
                       help='사전훈련 모델과 관찰공간 불일치 무시하고 새 모델 생성')
    
    # ✅ 랜덤성 강도 조정 옵션 추가
    parser.add_argument('--randomness_intensity', type=float, default=1.5,
                       help='훈련 시 랜덤성 강도 (0.0=없음, 1.0=기본, 2.0=강화, 기본값: 1.5)')
    
    return parser.parse_args()


def check_observation_compatibility(pretrained_model_path, current_env):
    """사전훈련 모델과 현재 환경의 관찰 공간 호환성 확인"""
    try:
        # 임시로 모델 로드해서 observation space 확인
        if os.path.exists(pretrained_model_path):
            temp_model = PPO.load(pretrained_model_path, env=None)
            
            # 모델의 observation space 추출
            if hasattr(temp_model.policy, 'observation_space'):
                model_obs_shape = temp_model.policy.observation_space.shape
            else:
                # 정책 네트워크의 첫 번째 레이어 크기로 추정
                first_layer = next(temp_model.policy.features_extractor.parameters())
                model_obs_shape = (first_layer.shape[1],)
            
            # 현재 환경의 observation space
            current_obs_shape = current_env.observation_space.shape
            
            print(f"🔍 관찰 공간 호환성 확인:")
            print(f"  사전훈련 모델: {model_obs_shape}")
            print(f"  현재 환경: {current_obs_shape}")
            
            compatible = model_obs_shape == current_obs_shape
            
            if compatible:
                print("✅ 관찰 공간 호환 가능")
            else:
                print("❌ 관찰 공간 불일치 감지")
                print("  옵션:")
                print("  1. --ignore_pretrained_obs_mismatch 플래그 사용")
                print("  2. 동일한 환경에서 훈련된 모델 사용")
                print("  3. 새로운 모델로 훈련 시작")
            
            del temp_model  # 메모리 정리
            return compatible
            
    except Exception as e:
        print(f"⚠️ 호환성 확인 실패: {e}")
        return False
    
    return False


def create_optimized_ppo_model(env, args, tensorboard_log=None):
    """2족 보행 최적화된 PPO 모델 생성"""
    
    def standing_lr_schedule(progress_remaining):
        if progress_remaining > 0.8:
            return 1e-4
        elif progress_remaining > 0.5:
            return 5e-5
        else:
            return 1e-5
            
    def clip_range_schedule(progress_remaining):
        if progress_remaining > 0.5:
            return 0.2
        else:
            return 0.1

    lr_schedule = standing_lr_schedule
    clip_range = clip_range_schedule
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=0.005,             # ✅ [수정] 초기 탐험을 장려하기 위해 엔트로피 계수 약간 증가 (기존 0.001)
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256], vf=[512, 256])],
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
            log_std_init=-2.0
        ),
        device='auto'
    )
    
    return model


class StandingTrainingCallback(BaseCallback):
    """2족 보행 특화 훈련 콜백 - 환경별 보상 객체 호환"""
    
    def __init__(self, args, eval_env, verbose=0):
        super().__init__(verbose)
        self.args = args
        self.eval_env = eval_env
        self.best_reward = -np.inf
        self.no_improvement_steps = 0
        self.patience = 1_000_000
        
        # 성능 추적
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=50)
        self.last_checkpoint = 0
        
        # 물구나무서기 통계 추적
        self.last_upside_down_count = 0
        
        # ⚠️ [제거] 전역 종료 원인 통계 추적 제거
        # self.termination_counts = defaultdict(int)

        # 정보 버퍼는 다른 용도로 사용될 수 있으므로 유지
        self.manual_info_buffer = deque(maxlen=args.num_envs * 2)
        
    def _get_reward_object(self, env):
        """환경에서 적절한 보상 객체 찾기"""
        if hasattr(env, 'bipedal_reward'):
            return env.bipedal_reward
        elif hasattr(env, 'standing_reward'):
            return env.standing_reward
        elif hasattr(env, 'env'):
            if hasattr(env.env, 'bipedal_reward'):
                return env.env.bipedal_reward
            elif hasattr(env.env, 'standing_reward'):
                return env.env.standing_reward
        return None
        
    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # ⚠️ [수정] 종료 원인 집계를 위해 버퍼를 채우는 로직은 유지하되,
        # 사용처가 없어졌으므로 향후 다른 통계에 활용될 수 있음.
        for i, done in enumerate(dones):
            if done:
                self.manual_info_buffer.append(copy.deepcopy(infos[i]))

        # 체크포인트 저장 로직
        if (self.num_timesteps - self.last_checkpoint >= self.args.checkpoint_interval):
            self._save_checkpoint()
            self.last_checkpoint = self.num_timesteps
            
        return True
    
    def _on_rollout_end(self) -> bool:
        """롤아웃 종료 시 통계 출력"""
        
        # ⚠️ [제거] 전역 종료 원인 집계 로직 제거
        
        # 버퍼는 비움
        self.manual_info_buffer.clear()
        
        # 기존 성능 평가 로직
        if len(self.locals.get('episode_rewards', [])) > 0:
            recent_rewards = self.locals['episode_rewards'][-10:]
            mean_reward = np.mean(recent_rewards)
            self.episode_rewards.extend(recent_rewards)
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.no_improvement_steps = 0
                self._save_best_model()
            else:
                self.no_improvement_steps += self.args.n_steps * self.args.num_envs
        
        # 기타 통계 수집 및 출력
        self._log_upside_down_statistics()
        
        # ⚠️ [제거] 종료 원인 통계 출력 함수 호출 제거
        
        # 조기 정지 확인
        if (self.args.early_stopping and 
            self.no_improvement_steps > self.patience):
            print(f"\n🛑 조기 정지: {self.patience:,} 스텝 동안 개선 없음")
            return False
            
        return True

    # ⚠️ [제거] _log_termination_statistics 메서드 전체 제거

    def _log_upside_down_statistics(self):
        """통계 로깅 - 환경별 보상 객체 호환"""
        try:
            # 환경에서 카운트 수집
            upside_down_counts = []
            
            # 모든 병렬 환경에서 통계 수집
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    try:
                        reward_obj = self._get_reward_object(env)
                        if reward_obj:
                            count = getattr(reward_obj, 'upside_down_count', 0)
                            upside_down_counts.append(count)
                    except:
                        pass
            
            # 통계 계산
            if upside_down_counts:
                total_upside_down = sum(upside_down_counts)
                new_attempts = total_upside_down - self.last_upside_down_count
                avg_per_env = total_upside_down / len(upside_down_counts)
                
                # PPO 로그와 함께 출력될 추가 정보
                print(f"🚨 물구나무서기 통계:")
                print(f"   총 시도 횟수: {total_upside_down}회")
                print(f"   이번 롤아웃 새로운 시도: {new_attempts}회")
                print(f"   환경당 평균: {avg_per_env:.1f}회")
                
                # TensorBoard에도 로깅
                if hasattr(self.logger, 'record'):
                    self.logger.record("custom/total_upside_down_attempts", total_upside_down)
                    self.logger.record("custom/new_upside_down_attempts", new_attempts)
                    self.logger.record("custom/avg_upside_down_per_env", avg_per_env)
                
                self.last_upside_down_count = total_upside_down
                
        except Exception as e:
            print(f"⚠️ 통계 수집 실패: {e}")
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_dir = Path("checkpoints") / f"{self.args.task}_training"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        
        # 통계도 메타데이터에 포함
        try:
            reward_obj = self._get_reward_object(self.eval_env)
            upside_down_count = getattr(reward_obj, 'upside_down_count', 0) if reward_obj else 0
        except:
            upside_down_count = 0
        
        metadata = {
            'timesteps': self.num_timesteps,
            'best_reward': self.best_reward,
            'upside_down_attempts': upside_down_count,
            'args': vars(self.args)
        }
        
        import json
        with open(checkpoint_dir / f"metadata_{self.num_timesteps}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"💾 체크포인트 저장: {checkpoint_path} (물구나무 시도: {upside_down_count}회)")
    
    def _save_best_model(self):
        """최고 성능 모델 저장"""
        best_dir = Path("models") / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        
        # 통계 포함
        try:
            reward_obj = self._get_reward_object(self.eval_env)
            upside_down_count = getattr(reward_obj, 'upside_down_count', 0) if reward_obj else 0
            upside_down_info = f" (물구나무: {upside_down_count}회)"
        except:
            upside_down_info = ""
        
        best_path = best_dir / f"{self.args.task}_best_{self.num_timesteps}.zip"
        self.model.save(best_path)
        print(f"🏆 최고 성능 모델 저장: {best_path} (보상: {self.best_reward:.2f}){upside_down_info}")

def train_with_optimized_parameters(args):  
    """2족 보행 최적화된 훈련 - 관찰 공간 호환성 수정"""
    print(f"\n🚀 2족 보행 최적화 훈련 시작! (task={args.task})")
    print(f"📊 최적화된 하이퍼파라미터:")
    print(f"  - 학습률: {args.learning_rate}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - 롤아웃 스텝: {args.n_steps}")
    print(f"  - 클립 범위: {args.clip_range}")
    print(f"  - 엔트로피: {args.entropy_coef}")
    print(f"  - 목표 속도: {args.target_vel} m/s")
    print(f"  - 안정성 가중치: {args.stability_weight}")
    print(f"  - 높이 허용오차: {args.height_tolerance}")
    print(f"  - 병렬 환경 수: {args.num_envs}")
    print(f"  - 총 훈련 스텝: {args.total_timesteps:,}")
    print(f"  - 커리큘럼 학습: {'사용' if args.use_curriculum else '미사용'}")
    print(f"  - 조기 정지: {'사용' if args.early_stopping else '미사용'}")
    
    # ✅ 랜덤성 강도 설정 추가
    
    # 명령행 인수에서 랜덤성 강도 가져오기
    randomness_intensity = args.randomness_intensity
    RobotPhysicsUtils.set_randomness_intensity(randomness_intensity)
    print(f"🎛️ 랜덤성 강도 설정: {randomness_intensity}")
    
    # 환경에 전달할 파라미터만 포함
    env_kwargs = {
        'randomize_physics': True,
    }
    
    # 환경 선택
    if args.task == "standing":
        if args.use_curriculum:
            env_class = GradualStandingEnv
            print("📚 점진적 커리큘럼 환경 사용")
        else:
            env_class = BipedalWalkingEnv
            print("🎯 2족 보행 환경 사용")
    else:
        env_class = Go1MujocoEnv
        print("🐕 기본 4족 보행 환경 사용")
    
    # 사전훈련 모델 호환성 확인
    use_pretrained = False
    compatible_env_kwargs = env_kwargs.copy()
    
    if args.pretrained_model:
        print(f"\n🔍 사전훈련 모델 호환성 확인 중...")
        
        # 모델 경로 확인
        pretrained_model_path = args.pretrained_model
        if pretrained_model_path == "latest":
            models = glob.glob(f"./models/{args.task}*.zip")
            if models:
                pretrained_model_path = list(sorted(models))[-1]
            else:
                print("❌ 'latest' 모델을 찾을 수 없습니다.")
                pretrained_model_path = None
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # 임시 환경 생성해서 관찰 공간 확인
            temp_env = env_class(**env_kwargs)
            is_compatible = check_observation_compatibility(pretrained_model_path, temp_env)
            temp_env.close()
            
            if is_compatible:
                use_pretrained = True
                print("✅ 사전훈련 모델 사용 가능")
            elif args.ignore_pretrained_obs_mismatch:
                print("⚠️ 관찰 공간 불일치를 무시하고 새 모델 생성")
                use_pretrained = False
            else:
                print("❌ 관찰 공간 불일치로 인해 사전훈련 모델 사용 불가")
                print("  해결책:")
                print("  1. --ignore_pretrained_obs_mismatch 플래그 추가")
                print("  2. 동일한 환경에서 훈련된 모델 사용")
                print("  3. 사전훈련 모델 없이 새로 시작")
                
                # 호환 모드로 환경 설정 시도
                print("  4. 호환 모드로 환경 설정 시도 중...")
                try:
                    compatible_env_kwargs['use_base_observation'] = True
                    temp_env_compat = env_class(**compatible_env_kwargs)
                    is_compatible_retry = check_observation_compatibility(pretrained_model_path, temp_env_compat)
                    temp_env_compat.close()
                    
                    if is_compatible_retry:
                        print("✅ 호환 모드로 설정 성공!")
                        use_pretrained = True
                        env_kwargs = compatible_env_kwargs  # 호환 모드 적용
                    else:
                        print("❌ 호환 모드로도 해결되지 않음")
                        # 사용자 선택 대기
                        choice = input("\n새 모델로 훈련을 계속하시겠습니까? (y/N): ").lower()
                        if choice != 'y':
                            print("훈련 중단")
                            return
                        use_pretrained = False
                except Exception as e:
                    print(f"⚠️ 호환 모드 설정 실패: {e}")
                    choice = input("\n새 모델로 훈련을 계속하시겠습니까? (y/N): ").lower()
                    if choice != 'y':
                        print("훈련 중단")
                        return
                    use_pretrained = False
        else:
            print(f"❌ 사전훈련 모델 파일을 찾을 수 없습니다: {pretrained_model_path}")
            use_pretrained = False
    
    # 학습용 환경 (병렬화) - 호환성 적용된 kwargs 사용
    print(f"\n🏭 {args.num_envs}개 병렬 환경 생성 중...")
    vec_env = make_vec_env(
        env_class, 
        n_envs=args.num_envs, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs  # 호환성 설정이 적용된 kwargs
    )
    
    # 평가용 환경 - 호환성 적용된 kwargs 사용
    print("📊 평가 환경 생성 중...")
    eval_env = env_class(render_mode="rgb_array", **env_kwargs)
    
    # 콜백 설정
    callbacks = [
        EnhancedVisualCallback(
            eval_env,
            eval_interval_minutes=args.visual_interval,
            n_eval_episodes=3,
            show_duration_seconds=args.show_duration,
            save_videos=args.save_videos,
            use_curriculum=args.use_curriculum
        ),
        StandingTrainingCallback(args, eval_env)
    ]
    
    # 비디오 녹화 콜백
    if args.video_interval > 0:
        record_env = DummyVecEnv([lambda: env_class(render_mode="rgb_array", **env_kwargs)])
        callbacks.append(
            VideoRecordingCallback(
                record_env,
                record_interval_timesteps=args.video_interval,
                video_folder=f"eval_videos_{args.task}",
                show_duration_seconds=args.show_duration
            )
        )
    
    # 모델 생성 또는 로드
    tensorboard_log = f"logs/{args.task}_optimized_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if use_pretrained:
        print(f"📂 사전 훈련 모델 로드: {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=vec_env)
        model.set_env(vec_env)
        
        # 하이퍼파라미터 업데이트
        if hasattr(model, 'learning_rate'):
            if args.use_curriculum:
                def lr_schedule(progress_remaining):
                    if progress_remaining > 0.9:
                        return args.learning_rate * 1.2
                    elif progress_remaining > 0.7:
                        return args.learning_rate
                    elif progress_remaining > 0.3:
                        return args.learning_rate * 0.5
                    else:
                        return args.learning_rate * 0.2
                model.learning_rate = lr_schedule
            else:
                model.learning_rate = args.learning_rate
        
        if hasattr(model, 'clip_range'):
            def clip_range_func(progress_remaining):
                return args.clip_range
            model.clip_range = clip_range_func
        
        print(f"✅ 하이퍼파라미터 업데이트 완료")
            
    else:
        print("🆕 새로운 모델 생성 중...")
        model = create_optimized_ppo_model(vec_env, args, tensorboard_log)
    
    # training_time 초기화
    training_time = 0.0
    
    # 학습 시작
    try:
        print(f"\n🎯 2족 보행 최적화 학습 시작...")
        print(f"📊 TensorBoard 로그: {tensorboard_log}")
        print("💡 TensorBoard 실행: tensorboard --logdir=logs")
        print("📈 실시간 모니터링을 위해 별도 터미널에서 TensorBoard를 실행하세요\n")
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if use_pretrained else True
        )
        
        training_time = time.time() - start_time
        print(f"\n⏱️ 총 훈련 시간: {training_time/3600:.2f}시간")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time if 'start_time' in locals() else 0.0
        print(f"\n⏹️ 사용자 중단 - 현재 상태 저장 중... (진행 시간: {training_time/3600:.2f}시간)")
    except Exception as e:
        training_time = time.time() - start_time if 'start_time' in locals() else 0.0
        print(f"\n❌ 오류 발생: {e}")
        print(f"⏱️ 진행 시간: {training_time/3600:.2f}시간")
        import traceback
        traceback.print_exc()
    
    # 최종 저장 및 분석
    print("\n💾 모델 및 결과 저장 중...")
    
    # 보고서 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"training_reports_{args.task}_optimized_{timestamp}"
    os.makedirs(report_path, exist_ok=True)
    
    if len(callbacks) > 0 and hasattr(callbacks[0], 'save_progress_report'):
        callbacks[0].save_progress_report(report_path)
        if hasattr(callbacks[0], 'save_detailed_analysis'):
            callbacks[0].save_detailed_analysis(report_path)
    
    # 최종 모델 저장
    model_path = f"models/{args.task}_optimized_final_{timestamp}.zip"
    Path("models").mkdir(exist_ok=True)
    model.save(model_path)
    print(f"✅ 최종 모델 저장: {model_path}")
    
    # 설정 저장
    config_path = os.path.join(report_path, "optimized_training_config.txt")
    with open(config_path, 'w') as f:
        f.write("=== 2족 보행 최적화 훈련 설정 ===\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Total timesteps: {args.total_timesteps:,}\n")
        f.write(f"Training time: {training_time/3600:.2f} hours\n")
        f.write(f"Used pretrained model: {use_pretrained}\n")
        f.write(f"Randomness intensity: {randomness_intensity}\n")  # ✅ 랜덤성 강도 기록
        if use_pretrained:
            f.write(f"Pretrained model path: {pretrained_model_path}\n")
        f.write(f"Environment observation mode: {'Base(45dim)' if env_kwargs.get('use_base_observation', False) else 'Extended(56dim)'}\n")
        f.write("\n")
        
        f.write("=== 환경 설정 ===\n")
        f.write(f"Environment class: {env_class.__name__}\n")
        f.write(f"Num environments: {args.num_envs}\n")
        f.write(f"Curriculum learning: {args.use_curriculum}\n")
        f.write(f"Target velocity: {args.target_vel} m/s\n")
        f.write(f"Stability weight: {args.stability_weight}\n")
        f.write(f"Height tolerance: {args.height_tolerance}\n")
        f.write(f"Use base observation: {env_kwargs.get('use_base_observation', False)}\n\n")
        
        f.write("=== PPO 하이퍼파라미터 ===\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"N steps: {args.n_steps}\n")
        f.write(f"Clip range: {args.clip_range}\n")
        f.write(f"Entropy coefficient: {args.entropy_coef}\n")
        f.write(f"Early stopping: {args.early_stopping}\n\n")
        
        f.write("=== 파일 경로 ===\n")
        f.write(f"Final model: {model_path}\n")
        f.write(f"TensorBoard logs: {tensorboard_log}\n")
        f.write(f"Training reports: {report_path}\n")
        
        if use_pretrained:
            f.write(f"Original pretrained model: {args.pretrained_model}\n")
            f.write(f"Observation compatibility: {'Compatible' if use_pretrained else 'Incompatible - created new model'}\n")
   
    # 최종 평가
    print(f"\n🧪 최종 모델 평가 중...")
    try:
        final_rewards = []
        final_successes = []
        
        for i in range(5):  # 5회 평가
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(1000):  # 최대 1000 스텝
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            final_rewards.append(episode_reward)
            # 환경에 따라 다른 성공 키 사용
            if hasattr(eval_env, 'bipedal_reward'):
                success_key = 'bipedal_success'
            else:
                success_key = 'standing_success'
            final_successes.append(info.get(success_key, False))
            print(f"  평가 {i+1}: 보상={episode_reward:.2f}, 길이={episode_length}, 성공={info.get(success_key, False)}")
        
        mean_reward = np.mean(final_rewards)
        success_rate = np.mean(final_successes)
        
        print(f"\n📊 최종 평가 결과:")
        print(f"  평균 보상: {mean_reward:.2f} ± {np.std(final_rewards):.2f}")
        print(f"  성공률: {success_rate:.1%}")
        
        # 결과를 config 파일에 추가
        with open(config_path, 'a') as f:
            f.write(f"\n=== 최종 평가 결과 ===\n")
            f.write(f"Mean reward: {mean_reward:.2f} ± {np.std(final_rewards):.2f}\n")
            f.write(f"Success rate: {success_rate:.1%}\n")
            
    except Exception as e:
        print(f"⚠️ 최종 평가 실패: {e}")
    
    # 정리
    eval_env.close()
    vec_env.close()
    if 'record_env' in locals():
        record_env.close()
    
    print(f"\n🎉 2족 보행 최적화 훈련 완료!")
    print(f"📁 결과 저장 위치: {report_path}")
    print(f"🎯 다음 단계:")
    print(f"   1. TensorBoard 확인: tensorboard --logdir=logs")
    print(f"   2. 보고서 확인: {report_path}")
    print(f"   3. 모델 테스트: python test_model.py --model {model_path}")
    print(f"   4. 비디오 확인: eval_videos_{args.task}/")
    
    if args.use_curriculum:
        print(f"   5. 커리큘럼 진행 상황은 TensorBoard에서 확인하세요")
    
    if not use_pretrained and args.pretrained_model:
        print(f"\n💡 참고: 관찰 공간 불일치로 인해 새 모델로 훈련했습니다.")
        print(f"   호환 가능한 사전훈련 모델을 사용하려면:")
        print(f"   1. 같은 환경 클래스에서 훈련된 모델 사용")
        print(f"   2. 또는 --ignore_pretrained_obs_mismatch 플래그 사용")
        print(f"   3. 또는 환경에 use_base_observation=True 설정")


if __name__ == "__main__":
    args = parse_arguments()
    train_with_optimized_parameters(args)