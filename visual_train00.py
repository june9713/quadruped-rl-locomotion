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
    from go1_standing_env import Go1StandingEnv, GradualStandingEnv
except ImportError:
    print("⚠️ go1_standing_env.py 파일이 필요합니다!")
    raise

# 한글 폰트 설정
font_name = 'Malgun Gothic'
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='강화학습 시각적 훈련')
    
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
    parser.add_argument('--total_timesteps', type=int, default=3_000_000,
                       help='총 훈련 스텝 수 (기본값: 3,000,000)')
    parser.add_argument('--num_envs', type=int, default=16,
                       help='병렬 환경 수 (기본값: 16)')
    parser.add_argument('--video_interval', type=int, default=100_000,
                       help='비디오 녹화 간격 (timesteps, 기본값: 100,000)')
    parser.add_argument('--use_curriculum', action='store_true',
                       help='커리큘럼 학습 사용')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='학습률 (기본값: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='배치 크기 (기본값: 256)')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='롤아웃 스텝 수 (기본값: 2048)')
    parser.add_argument('--clip_range', type=float, default=0.2,
                       help='PPO 클립 범위 (기본값: 0.2)')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='엔트로피 계수 (기본값: 0.01)')
    
    return parser.parse_args()


def create_ppo_model(env, args, tensorboard_log=None):
    """개선된 PPO 모델 생성"""
    
    # 커리큘럼 학습용 학습률 스케줄
    if args.use_curriculum:
        def lr_schedule(progress_remaining):
            # 초반에는 높은 학습률, 후반에는 낮은 학습률
            if progress_remaining > 0.8:
                return args.learning_rate
            elif progress_remaining > 0.5:
                return args.learning_rate * 0.5
            else:
                return args.learning_rate * 0.1
    else:
        lr_schedule = args.learning_rate
    
    # PPO 하이퍼파라미터
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        clip_range_vf=None,
        ent_coef=args.entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # 초반 탐색을 위해 True로 설정 가능
        sde_sample_freq=-1,
        target_kl=0.01,
        tensorboard_log=tensorboard_log,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 512, 256], vf=[512, 512, 256])],
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
        device='auto'
    )
    
    return model


def train_with_visual_feedback(args):
    """개선된 시각적 훈련"""
    print(f"\n🚀 개선된 시각적 훈련 시작! (task={args.task})")
    print(f"📊 하이퍼파라미터:")
    print(f"  - 학습률: {args.learning_rate}")
    print(f"  - 배치 크기: {args.batch_size}")
    print(f"  - 클립 범위: {args.clip_range}")
    print(f"  - 병렬 환경 수: {args.num_envs}")
    print(f"  - 커리큘럼 학습: {'사용' if args.use_curriculum else '미사용'}")
    
    # 환경 선택
    if args.task == "standing":
        if args.use_curriculum:
            env_class = GradualStandingEnv
        else:
            env_class = Go1StandingEnv
    else:
        env_class = Go1MujocoEnv
    
    # 학습용 환경 (병렬화)
    vec_env = make_vec_env(
        env_class, 
        n_envs=args.num_envs, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'randomize_physics': True}  # Domain randomization 활성화
    )
    
    # 평가용 환경
    eval_env = env_class(render_mode="rgb_array")
    
    # 콜백 설정
    callbacks = [
        EnhancedVisualCallback(  # 개선된 콜백 사용
            eval_env,
            eval_interval_minutes=args.visual_interval,
            n_eval_episodes=5,
            show_duration_seconds=args.show_duration,
            save_videos=args.save_videos,
            use_curriculum=args.use_curriculum
        )
    ]
    
    # 비디오 녹화 콜백
    print("args.video_interval"  ,args.video_interval)
    if args.video_interval > 0:
        record_env = DummyVecEnv([lambda: env_class(render_mode="rgb_array")])
        callbacks.append(
            VideoRecordingCallback(
                record_env,
                record_interval_timesteps=args.video_interval,
                video_folder=f"eval_videos_{args.task}",
                show_duration_seconds=args.show_duration
            )
        )
    
    # 모델 생성 또는 로드
    tensorboard_log = f"logs/{args.task}_enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.pretrained_model :
        modelPath = args.pretrained_model
        pretrained_model = modelPath
        if modelPath == "latest":
            models = glob.glob(f"./models/{args.task}*.zip")
            pretrained_model = list(sorted(models))[-1]
        elif os.path.exists(args.pretrained_model):
            #print(f"📂 사전 훈련 모델 로드: {args.pretrained_model}")
            pretrained_model = args.pretrained_model
        print(f"📂 사전 훈련 모델 로드: {pretrained_model}")
        model = PPO.load(pretrained_model, env=vec_env)
        model.set_env(vec_env)
    else:
        # torch import (PPO 내부에서 사용)
        try:
            import torch
        except ImportError:
            print("❌ PyTorch가 필요합니다. 설치해주세요: pip install torch")
            return
        
        model = create_ppo_model(vec_env, args, tensorboard_log)
    
    # 학습 시작
    try:
        print("\n🎯 학습 시작...")
        print(f"📊 TensorBoard 로그: {tensorboard_log}")
        print("💡 TensorBoard 실행: tensorboard --logdir=logs\n")
        
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if args.pretrained_model else True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # 최종 저장
    print("\n💾 모델 및 결과 저장 중...")
    
    # 보고서 저장
    report_path = f"training_reports_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(report_path, exist_ok=True)
    
    if len(callbacks) > 0 and hasattr(callbacks[0], 'save_progress_report'):
        callbacks[0].save_progress_report(report_path)
        callbacks[0].save_detailed_analysis(report_path)  # 추가 분석
    
    # 모델 저장
    model_path = f"models/{args.task}_enhanced_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    Path("models").mkdir(exist_ok=True)
    model.save(model_path)
    print(f"✅ 최종 모델 저장: {model_path}")
    
    # 훈련 설정 저장
    config_path = os.path.join(report_path, "training_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Task: {args.task}\n")
        f.write(f"Total timesteps: {args.total_timesteps:,}\n")
        f.write(f"Num environments: {args.num_envs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Clip range: {args.clip_range}\n")
        f.write(f"Entropy coefficient: {args.entropy_coef}\n")
        f.write(f"Curriculum learning: {args.use_curriculum}\n")
        f.write(f"Model saved at: {model_path}\n")
    
    # 정리
    eval_env.close()
    vec_env.close()
    if 'record_env' in locals():
        record_env.close()
    
    print(f"\n🎉 훈련 완료!")
    print(f"📁 결과 저장 위치: {report_path}")
    print(f"🎯 다음 단계:")
    print(f"   1. TensorBoard 확인: tensorboard --logdir=logs")
    print(f"   2. 보고서 확인: {report_path}")
    print(f"   3. 모델 테스트: python test_model.py --model {model_path}")


if __name__ == "__main__":
    args = parse_arguments()
    train_with_visual_feedback(args)