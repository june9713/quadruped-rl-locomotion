import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv
from tqdm import tqdm

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from go1_mujoco_env import Go1MujocoEnv
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList
from training_callback import VideoRecordingCallback # VisualTrainingCallback import 추가
from training_callback import VideoRecordingCallback, EnhancedVisualCallback

# training_callback.py 파일 상단에 추가해주세요.
import matplotlib.pyplot as plt
import platform

# OS에 맞는 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # MacOS
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')

# 마이너스 폰트 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

MODEL_DIR = "models"
LOG_DIR = "logs"


def train(args):
    vec_env = make_vec_env(
        Go1MujocoEnv,
        env_kwargs={"ctrl_type": args.ctrl_type, "biped": args.biped, "rand_power": args.rand_power},
        n_envs=args.num_parallel_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # 평가 및 비디오 녹화용 단일 환경 생성
    eval_record_env = Go1MujocoEnv(
        ctrl_type=args.ctrl_type,
        biped=args.biped,
        render_mode="rgb_array",
        camera_name="tracking",
        width=1024,
        height=768,
        rand_power=args.rand_power,
    )

    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name is None:
        run_name = f"{train_time}"
    else:
        run_name = f"{train_time}-{args.run_name}"

    model_path = f"{MODEL_DIR}/{run_name}"
    print(
        f"Training on {args.num_parallel_envs} parallel training environments and saving models to '{model_path}'"
    )

    # ✨ 수정된 부분: EnhancedVisualCallback의 평가 주기를 timestep 기반으로 변경
    # 비디오 녹화 주기(args.video_interval)와 동일하게 설정하여, 비디오가 녹화될 때마다
    # 평가, 모델 저장, 그래프 업데이트가 함께 수행되도록 합니다.
    enhanced_callback = EnhancedVisualCallback(
        eval_env=eval_record_env,
        best_model_save_path=model_path,
        eval_freq=args.video_interval,  # 분(minutes) 단위 대신 timestep 단위로 변경
        n_eval_episodes=3,
        show_duration_seconds=20,
        save_videos=True, # 평가 중 비디오 저장 활성화
    )
    
    # 훈련 중 주기적인 비디오 녹화 콜백 (step 기반)
    video_callback = VideoRecordingCallback(
        record_env=eval_record_env,
        record_interval_timesteps=args.video_interval,
        show_duration_seconds=args.video_duration
    )
    
    # ✨ 추가된 부분: rand_power 커리큘럼 콜백
    # 학습 진행도에 따라 rand_power를 동적으로 조절합니다.
    from training_callback import CurriculumCallback
    curriculum_callback = CurriculumCallback(
        total_timesteps=args.total_timesteps,
        initial_rand_power=args.rand_power
    )

    # 세 개의 콜백을 함께 사용
    callback_list = CallbackList([enhanced_callback, video_callback, curriculum_callback])

    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, env=vec_env, verbose=1, tensorboard_log=LOG_DIR
        )
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)

    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        progress_bar=True,
        tb_log_name=run_name,
        callback=callback_list,
    )
    # 최종 모델 저장
    model.save(f"{model_path}/final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=12,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=5,
        help="Number of episodes to test the model",
    )
    parser.add_argument(
        "--record_test_episodes",
        action="store_true",
        help="Whether to record the test episodes or not. If false, the episodes are rendered in the window.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=5_000_000,
        help="Number of timesteps to train the model for",
    )
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10_000,
        help="The frequency of evaluating the models while training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="position",
        help="Whether the model should control the robot using torque or position control.",
    )
    parser.add_argument(
        "--biped",
        action="store_true",
        help="If set, the robot will be trained for bipedal walking.",
    )

    #video_duration
    parser.add_argument(
        "--video_duration",
        type=int,
        default=10,
        help="Duration of the video to record",
    )

    parser.add_argument(
        "--video_interval",
        type=int,
        default=300_000,
        help="Number of timesteps interval to record the video",
    )
    
    # ✨ 추가된 부분: 랜덤 강도 인자
    parser.add_argument(
        "--rand_power",
        type=float,
        default=0.0,
        help="Amount of randomization to apply to joint angles at reset. 0.0 means no randomization.",
    )

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args)