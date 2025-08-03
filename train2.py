#!/usr/bin/env python3
"""
실시간 시각적 피드백과 함께하는 훈련 스크립트
"""

import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv

def main():
    parser = argparse.ArgumentParser(description="시각적 피드백과 함께하는 훈련")
    parser.add_argument("--task", choices=["walking", "handstand"], default="walking")
    parser.add_argument("--pretrained_model", type=str, help="기존 모델 경로")
    parser.add_argument("--visual_interval", type=int, default=10, 
                       help="시각화 간격 (분)")
    parser.add_argument("--show_duration", type=int, default=30,
                       help="시뮬레이션 보여주는 시간 (초)")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--save_videos", action="store_true", 
                       help="비디오 저장 여부")
    
    args = parser.parse_args()
    
    print(f"🚀 {args.task} 훈련 시작!")
    print(f"📊 {args.visual_interval}분마다 {args.show_duration}초간 시각화")
    
    # 환경 설정
    if args.task == "handstand":
        env_class = Go1HandstandEnv
        model_save_path = "models/handstand_visual"
    else:
        env_class = Go1MujocoEnv
        model_save_path = "models/walking_visual"
    
    # 훈련 환경
    vec_env = make_vec_env(
        env_class,
        n_envs=args.num_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv
    )
    
    # 시각화용 환경
    eval_env = env_class(render_mode="human")
    
    # 시각적 콜백 생성
    visual_callback = VisualTrainingCallback(
        eval_env=eval_env,
        eval_interval_minutes=args.visual_interval,
        n_eval_episodes=3,
        show_duration_seconds=args.show_duration,
        save_videos=args.save_videos
    )
    
    callbacks = [visual_callback]
    
    # 비디오 녹화 추가
    if args.save_videos:
        video_callback = VideoRecordingCallback(
            eval_env=eval_env,
            record_interval_timesteps=200_000,
            video_folder=f"{model_save_path}/videos"
        )
        callbacks.append(video_callback)
    
    # 모델 생성/로드
    if args.pretrained_model:
        print(f"📂 기존 모델 로드: {args.pretrained_model}")
        model = PPO.load(args.pretrained_model, env=vec_env)
    else:
        print("🆕 새로운 모델 생성")
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1,
            tensorboard_log="logs/visual_training",
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048
        )
    
    os.makedirs(model_save_path, exist_ok=True)
    
    try:
        print(f"🎯 총 {args.total_timesteps:,} timesteps 훈련 시작!")
        print("💡 훈련 중 실시간 시뮬레이션이 주기적으로 표시됩니다.")
        print("💡 그래프 창이 열려서 실시간 성능을 보여줍니다.")
        
        # 훈련 시작!
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"{args.task}_visual_training"
        )
        
        print("✅ 훈련 완료!")
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 훈련을 중단했습니다.")
        
    finally:
        # 최종 모델 저장
        final_model_path = f"{model_save_path}/final_model"
        model.save(final_model_path)
        print(f"💾 최종 모델 저장: {final_model_path}")
        
        # 진행 상황 보고서 저장
        visual_callback.save_progress_report(model_save_path)
        
        # 환경 정리
        eval_env.close()
        vec_env.close()
        
        print("🎉 모든 작업 완료!")
        print(f"📊 보고서 위치: {model_save_path}/training_progress.png")
        if args.save_videos:
            print(f"🎥 비디오 위치: {model_save_path}/videos/")

if __name__ == "__main__":
    main()