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
import traceback
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
    from go1_standing_env import Go1StandingEnv, GradualStandingEnv, BipedalWalkingEnv, BipedalCurriculumEnv, RobotPhysicsUtils
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
    
    parser.add_argument('--extreme_gpu', action='store_true',
                       help='GPU 활용을 극대화하는 하이퍼파라미터 및 최적화 적용')

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
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='학습률 (기본값: 2e-4)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='배치 크기 (기본값: 128)')
    parser.add_argument('--n_steps', type=int, default=1024,
                       help='롤아웃 스텝 수 (기본값: 1024)')
    parser.add_argument('--clip_range', type=float, default=0.15,
                       help='PPO 클립 범위 (기본값: 0.15)')
    parser.add_argument('--entropy_coef', type=float, default=0.005,
                       help='엔트로피 계수 (기본값: 0.005)')
    
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
    
    parser.add_argument('--ignore_pretrained_obs_mismatch', action='store_true',
                       help='사전훈련 모델과 관찰공간 불일치 무시하고 새 모델 생성')
    
    parser.add_argument('--randomness_intensity', type=float, default=1.5,
                       help='훈련 시 랜덤성 강도 (0.0=없음, 1.0=기본, 2.0=강화, 기본값: 1.5)')
    
    return parser.parse_args()


def check_observation_compatibility(pretrained_model_path, current_env):
    """사전훈련 모델과 현재 환경의 관찰 공간 호환성 확인"""
    try:
        if os.path.exists(pretrained_model_path):
            # --- 수정: 호환성 확인 로직 간소화 ---
            # 여기서 발생하는 state_dict 오류는 무시하고, 실제 로드 시 처리하도록 함
            # 여기서는 관찰 공간 크기만 비교하는 것이 목적
            from stable_baselines3.common.save_util import load_from_zip_file
            data, params, pytorch_variables = load_from_zip_file(pretrained_model_path)
            
            # 모델의 관찰 공간 shape 추정
            # policy_kwargs가 저장되어 있다면 그것을 사용
            if params and 'policy' in params and 'observation_space' in params['policy']:
                 model_obs_shape = params['policy']['observation_space'].shape
            else:
                # 없다면, state_dict에서 첫 번째 레이어 크기로 추정 (부정확할 수 있음)
                first_weight_key = next(iter(data['policy']))
                first_weight_tensor = data['policy'][first_weight_key]
                # This is a heuristic and might not always be correct
                model_obs_shape = (first_weight_tensor.shape[1],) if len(first_weight_tensor.shape) > 1 else None

            current_obs_shape = current_env.observation_space.shape
            
            print(f"🔍 관찰 공간 호환성 확인:")
            print(f"  사전훈련 모델 (추정): {model_obs_shape}")
            print(f"  현재 환경: {current_obs_shape}")

            if model_obs_shape is None or model_obs_shape[0] != current_obs_shape[0]:
                 print("❌ 관찰 공간 불일치 감지")
                 return False
            else:
                 print("✅ 관찰 공간 호환 가능")
                 return True
            
    except Exception as e:
        print(f"⚠️ 호환성 확인 중 오류 발생 (로드 시 재시도): {e}")
        # 확인 단계에서 오류가 나더라도, 실제 로드 로직에서 처리할 수 있으므로 True를 반환하여 진행
        return True
    
    return False


def create_optimized_ppo_model(env, args, device, tensorboard_log=None):
    """2족 보행 최적화된 PPO 모델 생성"""
    
    if args.extreme_gpu:
        print("🚀 극단적 GPU 활용 모드로 PPO 모델을 생성합니다.")
        ppo_params = {
            'n_steps': 8192,
            'batch_size': 1024,
            'n_epochs': 60,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'learning_rate': 3e-5,
            'clip_range': 0.2,
            'ent_coef': 0.001,
            'vf_coef': 0.5,
            'max_grad_norm': 1.0,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
                log_std_init=-2.0
            ),
        }
    else:
        def standing_lr_schedule(progress_remaining):
            if progress_remaining > 0.8: return 3e-5
            elif progress_remaining > 0.5: return 2e-5
            else: return 1e-5
                
        def clip_range_schedule(progress_remaining):
            return 0.2 if progress_remaining > 0.5 else 0.1

        ppo_params = {
            'n_steps': 4096,
            'batch_size': 256,
            'n_epochs': 30,
            'gamma': 0.98,
            'gae_lambda': 0.95,
            'learning_rate': standing_lr_schedule,
            'clip_range': clip_range_schedule,
            'ent_coef': 0.005,
            'vf_coef': 0.7,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[512, 256], vf=[512, 256])],
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
                log_std_init=-2.0
            ),
        }

    model = PPO(
        "MlpPolicy",
        env,
        normalize_advantage=True,
        use_sde=False,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=device,
        **ppo_params
    )
    
    return model

# StandingTrainingCallback 클래스는 변경 사항 없음 (생략)
class StandingTrainingCallback(BaseCallback):
    def __init__(self, args, eval_env, verbose=0):
        super().__init__(verbose)
        self.args = args
        self.eval_env = eval_env
        self.best_reward = -np.inf
        self.no_improvement_steps = 0
        self.patience = 1_000_000
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=50)
        self.last_checkpoint = 0
        self.last_upside_down_count = 0
        self.manual_info_buffer = deque(maxlen=args.num_envs * 2)
    def _get_reward_object(self, env):
        if hasattr(env, 'bipedal_reward'): return env.bipedal_reward
        elif hasattr(env, 'standing_reward'): return env.standing_reward
        elif hasattr(env, 'env'):
            if hasattr(env.env, 'bipedal_reward'): return env.env.bipedal_reward
            elif hasattr(env.env, 'standing_reward'): return env.env.standing_reward
        return None
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done: self.manual_info_buffer.append(copy.deepcopy(infos[i]))
        if (self.num_timesteps - self.last_checkpoint >= self.args.checkpoint_interval):
            self._save_checkpoint()
            self.last_checkpoint = self.num_timesteps
        return True
    def _on_rollout_end(self) -> bool:
        self.manual_info_buffer.clear()
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
        self._log_upside_down_statistics()
        if (self.args.early_stopping and self.no_improvement_steps > self.patience):
            print(f"\n🛑 조기 정지: {self.patience:,} 스텝 동안 개선 없음")
            return False
        return True
    def _log_upside_down_statistics(self):
        try:
            upside_down_counts = []
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    try:
                        reward_obj = self._get_reward_object(env)
                        if reward_obj: upside_down_counts.append(getattr(reward_obj, 'upside_down_count', 0))
                    except: pass
            if upside_down_counts:
                total_upside_down = sum(upside_down_counts)
                new_attempts = total_upside_down - self.last_upside_down_count
                avg_per_env = total_upside_down / len(upside_down_counts)
                print(f"🚨 물구나무서기 통계: 총 {total_upside_down}회, 이번 롤아웃 {new_attempts}회, 환경당 평균 {avg_per_env:.1f}회")
                if hasattr(self.logger, 'record'):
                    self.logger.record("custom/total_upside_down_attempts", total_upside_down)
                    self.logger.record("custom/new_upside_down_attempts", new_attempts)
                    self.logger.record("custom/avg_upside_down_per_env", avg_per_env)
                self.last_upside_down_count = total_upside_down
        except Exception as e: print(f"⚠️ 통계 수집 실패: {e}")
    def _save_checkpoint(self):
        checkpoint_dir = Path("checkpoints") / f"{self.args.task}_training"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        try: reward_obj = self._get_reward_object(self.eval_env); upside_down_count = getattr(reward_obj, 'upside_down_count', 0) if reward_obj else 0
        except: upside_down_count = 0
        metadata = {'timesteps': self.num_timesteps, 'best_reward': self.best_reward, 'upside_down_attempts': upside_down_count, 'args': vars(self.args)}
        import json
        with open(checkpoint_dir / f"metadata_{self.num_timesteps}.json", 'w') as f: json.dump(metadata, f, indent=2)
        print(f"💾 체크포인트 저장: {checkpoint_path} (물구나무 시도: {upside_down_count}회)")
    def _save_best_model(self):
        best_dir = Path("models") / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        try: reward_obj = self._get_reward_object(self.eval_env); upside_down_count = getattr(reward_obj, 'upside_down_count', 0) if reward_obj else 0; upside_down_info = f" (물구나무: {upside_down_count}회)"
        except: upside_down_info = ""
        best_path = best_dir / f"{self.args.task}_best_{self.num_timesteps}.zip"
        self.model.save(best_path)
        print(f"🏆 최고 성능 모델 저장: {best_path} (보상: {self.best_reward:.2f}){upside_down_info}")


def load_compiled_model(model_path, env, device):
    """torch.compile로 저장된 모델을 안전하게 불러오는 함수"""
    print(f"📦 컴파일된 모델 체크포인트 로드를 시도합니다: {model_path}")
    
    from stable_baselines3.common.save_util import load_from_zip_file
    
    # 1. 체크포인트 파일에서 데이터를 먼저 로드합니다.
    data, params, pytorch_variables = load_from_zip_file(model_path, device=device)
    
    # 2. 모델의 하이퍼파라미터로 새로운 PPO 모델을 생성합니다.
    #    이렇게 하면 올바른 신경망 구조를 가진 모델이 만들어집니다.
    model = PPO(
        policy=params["policy_class"],
        env=env,
        device=device,
        _init_setup_model=False, # 모델을 바로 초기화하지 않음
    )
    model.set_parameters(params, exact_match=False) # 로드된 파라미터 적용
    model._setup_model() # 모델 초기화

    # 3. state_dict의 키에서 '_orig_mod.' 접두사를 제거합니다.
    policy_state_dict = data['policy']
    cleaned_state_dict = {}
    for k, v in policy_state_dict.items():
        cleaned_state_dict[k.replace("_orig_mod.", "")] = v
        
    # 4. 정리된 state_dict를 모델 정책에 로드합니다.
    model.policy.load_state_dict(cleaned_state_dict)
    
    print("✅ 컴파일된 모델 상태 복구 및 로드 성공!")
    return model


def train_with_optimized_parameters(args):
    """2족 보행 최적화 훈련 - 관찰 공간 호환성 자동 확인 및 수정 적용"""
    
    if torch.cuda.is_available():
        print("✅ CUDA 사용 가능. GPU(RTX 5080)로 훈련을 시작합니다.")
        print(f"   - PyTorch 버전: {torch.__version__}")
        print(f"   - CUDA 버전: {torch.version.cuda}")
        print(f"   - GPU 장치: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        torch.set_float32_matmul_precision('high')
    else:
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 훈련을 시작합니다.")
        device = torch.device("cpu")

    if args.extreme_gpu:
        print("="*60)
        print("⚠️ '극단적 GPU 활용 모드'가 활성화되었습니다.")
        print("   - 병렬 환경 수(--num_envs)를 CPU 코어 수에 맞게 높여주세요 (예: 16, 24, 32).")
        print("   - GPU 메모리 사용량이 크게 증가할 수 있습니다.")
        print("="*60)

    print(f"\n🚀 2족 보행 최적화 훈련 시작! (task={args.task})")
    
    training_time = 0.0
    randomness_intensity = args.randomness_intensity
    RobotPhysicsUtils.set_randomness_intensity(args.randomness_intensity)
    print(f"🎛️ 랜덤성 강도 설정: {args.randomness_intensity}")

    env_class = BipedalWalkingEnv if args.task == "standing" else Go1MujocoEnv
    env_kwargs = {'randomize_physics': True}
    print(f"🎯 훈련 환경: {env_class.__name__}")

    use_pretrained = False
    pretrained_model_path = args.pretrained_model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        use_pretrained = True

    print(f"\n🏭 {args.num_envs}개 병렬 환경 생성 중...")
    vec_env = make_vec_env(env_class, n_envs=args.num_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    eval_env = env_class(render_mode="rgb_array", **env_kwargs)

    callbacks = [
        EnhancedVisualCallback(eval_env, eval_interval_minutes=args.visual_interval, n_eval_episodes=3, show_duration_seconds=args.show_duration, save_videos=args.save_videos),
        StandingTrainingCallback(args, eval_env)
    ]
    if args.video_interval > 0:
        record_env = DummyVecEnv([lambda: env_class(render_mode="rgb_array", **env_kwargs)])
        callbacks.append(
            VideoRecordingCallback(record_env, record_interval_timesteps=args.video_interval, video_folder=f"eval_videos_{args.task}", show_duration_seconds=args.show_duration)
        )

    tensorboard_log = f"logs/{args.task}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if use_pretrained:
        print(f"📂 사전 훈련 모델 로드를 시도합니다: {pretrained_model_path}")
        try:
            # --- 수정: torch.compile 복구 로직 적용 ---
            # 먼저 일반 로드를 시도하고, 실패하면 컴파일된 모델 복구 로직을 실행합니다.
            try:
                model = PPO.load(pretrained_model_path, env=vec_env, device=device)
                print("✅ 모델 일반 로드 성공.")
            except Exception:
                model = load_compiled_model(pretrained_model_path, vec_env, device)
            
        except Exception as e:
            print(f"❌ 모델 로드에 최종 실패했습니다: {e}")
            print("🆕 새로운 모델로 훈련을 시작합니다.")
            model = create_optimized_ppo_model(vec_env, args, device, tensorboard_log)
    else:
        print("🆕 새로운 모델 생성 중...")
        model = create_optimized_ppo_model(vec_env, args, device, tensorboard_log)
    
    if args.extreme_gpu and device.type == 'cuda' and hasattr(torch, 'compile'):
        print("🚀 PyTorch 2.x JIT 컴파일러(torch.compile)를 정책 모델에 적용합니다...")
        try:
            model.policy = torch.compile(model.policy)
            print("✅ torch.compile 적용 성공!")
        except Exception as e:
            print(f"⚠️ torch.compile 적용 실패. Triton이 설치되지 않았거나 지원되지 않는 환경(예: Windows)일 수 있습니다.")
            print(f"   (오류: {e})")
            print(f"   JIT 컴파일 없이 훈련을 계속합니다.")

    try:
        start_time = time.time()
        print(f"\n🎯 학습 시작...")
        # reset_num_timesteps=False로 설정하여 이전 학습 스텝을 이어가도록 함
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not use_pretrained 
        )
        training_time = time.time() - start_time
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        traceback.print_exc()

    # 최종 저장 및 분석 (생략)
    print("\n💾 모델 및 결과 저장 중...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"training_reports_{args.task}_optimized_{timestamp}"
    os.makedirs(report_path, exist_ok=True)
    
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
    
    # ✅ [수정] --use_curriculum 플래그에 따라 환경을 선택하도록 변경
    if args.use_curriculum:
        # 커리큘럼 플래그가 있으면 BipedalCurriculumEnv 사용
        env_class = BipedalCurriculumEnv 
        print("🎓 커리큘럼 모드로 훈련을 시작합니다. (BipedalCurriculumEnv)")
    else:
        # 기본 모드
        env_class = BipedalWalkingEnv if args.task == "standing" else Go1MujocoEnv

    env_kwargs = {'randomize_physics': True}
    print(f"🎯 훈련 환경: {env_class.__name__}")
    
    if not use_pretrained and args.pretrained_model:
        print(f"\n💡 참고: 관찰 공간 불일치로 인해 새 모델로 훈련했습니다.")
        print(f"   호환 가능한 사전훈련 모델을 사용하려면:")
        print(f"   1. 같은 환경 클래스에서 훈련된 모델 사용")
        print(f"   2. 또는 --ignore_pretrained_obs_mismatch 플래그 사용")
        print(f"   3. 또는 환경에 use_base_observation=True 설정")

if __name__ == "__main__":
    args = parse_arguments()
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    train_with_optimized_parameters(args)
