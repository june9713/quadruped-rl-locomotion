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
try:
    from go1_standing_env import Go1StandingEnv, GradualStandingEnv, BipedalWalkingEnv
except ImportError:
    print("⚠️ go1_standing_env.py 파일이 필요합니다!")
    raise

# 한글 폰트 설정
font_name = 'Malgun Gothic'
plt.rc('font', family=font_name)
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
    parser.add_argument('--total_timesteps', type=int, default=5_000_000,  # ✅ 증가
                       help='총 훈련 스텝 수 (기본값: 5,000,000)')
    parser.add_argument('--num_envs', type=int, default=12,  # ✅ 약간 축소 (안정성)
                       help='병렬 환경 수 (기본값: 12)')
    parser.add_argument('--video_interval', type=int, default=150_000,  # ✅ 더 자주
                       help='비디오 녹화 간격 (timesteps, 기본값: 150,000)')
    parser.add_argument('--use_curriculum', action='store_true',
                       help='커리큘럼 학습 사용')
    
    # ✅ 2족 보행 최적화된 하이퍼파라미터
    parser.add_argument('--learning_rate', type=float, default=2e-4,  # ✅ 축소 (안정성)
                       help='학습률 (기본값: 2e-4)')
    parser.add_argument('--batch_size', type=int, default=128,  # ✅ 축소 (안정성)
                       help='배치 크기 (기본값: 128)')
    parser.add_argument('--n_steps', type=int, default=1024,  # ✅ 축소 (빈번한 업데이트)
                       help='롤아웃 스텝 수 (기본값: 1024)')
    parser.add_argument('--clip_range', type=float, default=0.15,  # ✅ 축소 (안정적 업데이트)
                       help='PPO 클립 범위 (기본값: 0.15)')
    parser.add_argument('--entropy_coef', type=float, default=0.005,  # ✅ 축소 (덜 탐험적)
                       help='엔트로피 계수 (기본값: 0.005)')
    
    # ✅ 새로운 2족 보행 특화 파라미터
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
    
    return parser.parse_args()


def create_optimized_ppo_model(env, args, tensorboard_log=None):
    """2족 보행 최적화된 PPO 모델 생성 - clip_range 오류 수정"""
    
    # ✅ 2족 보행용 학습률 스케줄
    def standing_lr_schedule(progress_remaining):
        """2족 보행 최적화 학습률 스케줄"""
        if progress_remaining > 0.9:
            # 초기: 빠른 학습
            return args.learning_rate * 1.2
        elif progress_remaining > 0.7:
            # 중기: 안정적 학습
            return args.learning_rate
        elif progress_remaining > 0.3:
            # 후기: 세밀한 조정
            return args.learning_rate * 0.5
        else:
            # 마지막: 매우 세밀한 조정
            return args.learning_rate * 0.2
    
    # ✅ 수정: clip_range도 함수로 설정
    def clip_range_schedule(progress_remaining):
        """클립 범위 스케줄"""
        return args.clip_range
    
    if args.use_curriculum:
        lr_schedule = standing_lr_schedule
        clip_range = clip_range_schedule
    else:
        lr_schedule = args.learning_rate
        clip_range = clip_range_schedule  # ✅ 항상 함수로 설정
    
    # ✅ 2족 보행 최적화 PPO 하이퍼파라미터
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=8,  # 축소 (과적합 방지)
        gamma=0.995,  # 증가 (장기 안정성 중시)
        gae_lambda=0.98,  # 증가 (안정성)
        clip_range=clip_range,  # ✅ 함수로 설정
        clip_range_vf=0.2,  # value function 클리핑
        ent_coef=args.entropy_coef,
        vf_coef=0.8,  # 증가 (value function 중시)
        max_grad_norm=0.3,  # 축소 (gradient 안정성)
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.015,  # 증가 (안정적 업데이트)
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # 네트워크 축소
            activation_fn=torch.nn.Tanh,  # Tanh 사용 (안정적)
            ortho_init=True,
            log_std_init=-0.5,  # 초기 탐험 축소
        ),
        device='auto'
    )
    
    return model


class StandingTrainingCallback(BaseCallback):
    """2족 보행 특화 훈련 콜백"""
    
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
        
        # ✅ 물구나무서기 통계 추적
        self.last_upside_down_count = 0
        
    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 체크포인트 저장
        if (self.num_timesteps - self.last_checkpoint >= self.args.checkpoint_interval):
            self._save_checkpoint()
            self.last_checkpoint = self.num_timesteps
            
        return True
    
    def _on_rollout_end(self) -> bool:
        """✅ 롤아웃 종료 시 물구나무서기 통계도 함께 출력"""
        
        # 기존 성능 평가 로직
        if len(self.locals.get('episode_rewards', [])) > 0:
            recent_rewards = self.locals['episode_rewards'][-10:]
            mean_reward = np.mean(recent_rewards)
            self.episode_rewards.extend(recent_rewards)
            
            # 개선 추적
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.no_improvement_steps = 0
                self._save_best_model()
            else:
                self.no_improvement_steps += self.args.n_steps * self.args.num_envs
        
        # ✅ 물구나무서기 통계 수집 및 출력
        self._log_upside_down_statistics()
        
        # 조기 정지 확인
        if (self.args.early_stopping and 
            self.no_improvement_steps > self.patience):
            print(f"\n🛑 조기 정지: {self.patience:,} 스텝 동안 개선 없음")
            return False
            
        return True


    def _log_upside_down_statistics(self):
        """✅ 물구나무서기 통계 로깅"""
        try:
            # 환경에서 물구나무서기 카운트 수집
            upside_down_counts = []
            
            # 모든 병렬 환경에서 통계 수집
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    try:
                        # 환경의 standing_reward에서 카운트 가져오기
                        if hasattr(env, 'standing_reward'):
                            count = getattr(env.standing_reward, 'upside_down_count', 0)
                            upside_down_counts.append(count)
                        elif hasattr(env, 'env') and hasattr(env.env, 'standing_reward'):
                            # Wrapper가 있는 경우
                            count = getattr(env.env.standing_reward, 'upside_down_count', 0)
                            upside_down_counts.append(count)
                    except:
                        pass
            
            # 통계 계산
            if upside_down_counts:
                total_upside_down = sum(upside_down_counts)
                new_attempts = total_upside_down - self.last_upside_down_count
                avg_per_env = total_upside_down / len(upside_down_counts)
                
                # ✅ PPO 로그와 함께 출력될 추가 정보
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
            print(f"⚠️ 물구나무서기 통계 수집 실패: {e}")
    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_dir = Path("checkpoints") / f"{self.args.task}_training"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.num_timesteps}.zip"
        self.model.save(checkpoint_path)
        
        # ✅ 물구나무서기 통계도 메타데이터에 포함
        try:
            upside_down_count = getattr(self.eval_env.standing_reward, 'upside_down_count', 0)
        except:
            upside_down_count = 0
        
        metadata = {
            'timesteps': self.num_timesteps,
            'best_reward': self.best_reward,
            'upside_down_attempts': upside_down_count,  # ✅ 추가
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
        
        # ✅ 물구나무서기 통계 포함
        try:
            upside_down_count = getattr(self.eval_env.standing_reward, 'upside_down_count', 0)
            upside_down_info = f" (물구나무: {upside_down_count}회)"
        except:
            upside_down_info = ""
        
        best_path = best_dir / f"{self.args.task}_best_{self.num_timesteps}.zip"
        self.model.save(best_path)
        print(f"🏆 최고 성능 모델 저장: {best_path} (보상: {self.best_reward:.2f}){upside_down_info}")


def train_with_optimized_parameters(args):  
    """2족 보행 최적화된 훈련 - 오류 수정 버전"""
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
    
    # ✅ 수정: 환경에 전달할 파라미터만 포함
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
    
    # 학습용 환경 (병렬화)
    print(f"\n🏭 {args.num_envs}개 병렬 환경 생성 중...")
    vec_env = make_vec_env(
        env_class, 
        n_envs=args.num_envs, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )
    
    # 평가용 환경
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
        StandingTrainingCallback(args, eval_env)  # ✅ 개선된 콜백 사용
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
    
    if args.pretrained_model:
        modelPath = args.pretrained_model
        pretrained_model = modelPath
        if modelPath == "latest":
            models = glob.glob(f"./models/{args.task}*.zip")
            pretrained_model = list(sorted(models))[-1]
        elif os.path.exists(args.pretrained_model):
            pretrained_model = args.pretrained_model
        
        print(f"📂 사전 훈련 모델 로드: {pretrained_model}")
        model = PPO.load(pretrained_model, env=vec_env)
        model.set_env(vec_env)
        
        # ✅ 수정: clip_range 함수로 설정 (float 오류 해결)
        if hasattr(model, 'learning_rate'):
            if args.use_curriculum:
                # 커리큘럼 학습률 스케줄
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
        
        # ✅ 수정: clip_range를 함수로 설정
        if hasattr(model, 'clip_range'):
            def clip_range_func(progress_remaining):
                return args.clip_range
            model.clip_range = clip_range_func
        
        print(f"✅ 하이퍼파라미터 업데이트 완료")
            
    else:
        print("🆕 새로운 모델 생성 중...")
        model = create_optimized_ppo_model(vec_env, args, tensorboard_log)
    
    # ✅ 수정: training_time 초기화
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
            reset_num_timesteps=False if args.pretrained_model else True
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
    
    # ✅ 수정: training_time 안전하게 사용
    config_path = os.path.join(report_path, "optimized_training_config.txt")
    with open(config_path, 'w') as f:
        f.write("=== 2족 보행 최적화 훈련 설정 ===\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Total timesteps: {args.total_timesteps:,}\n")
        f.write(f"Training time: {training_time/3600:.2f} hours\n\n")
        
        f.write("=== 환경 설정 ===\n")
        f.write(f"Num environments: {args.num_envs}\n")
        f.write(f"Curriculum learning: {args.use_curriculum}\n")
        f.write(f"Target velocity: {args.target_vel} m/s (설정값, 환경 내부에서 0.0 사용)\n")
        f.write(f"Stability weight: {args.stability_weight} (설정값, 환경 내부에서 고정값 사용)\n")
        f.write(f"Height tolerance: {args.height_tolerance} (설정값, 환경 내부에서 0.15 사용)\n\n")
        
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
        
        # ✅ 수정: 사전 훈련 모델 정보 추가
        if args.pretrained_model:
            f.write(f"Pretrained model: {args.pretrained_model}\n")
            f.write("Note: clip_range was converted to function to fix compatibility\n")
    
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
            final_successes.append(info.get('standing_success', False))
            print(f"  평가 {i+1}: 보상={episode_reward:.2f}, 길이={episode_length}, 성공={info.get('standing_success', False)}")
        
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



if __name__ == "__main__":
    args = parse_arguments()
    train_with_optimized_parameters(args)