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

# ✨ --- 수정된 부분 시작 --- ✨
# Tcl/Tk 오류를 방지하기 위해 GUI 백엔드 대신 Agg 백엔드를 사용하도록 설정합니다.
# 이 코드는 matplotlib.pyplot을 import하기 전에 실행되어야 합니다.
import matplotlib
matplotlib.use('Agg')
# ✨ --- 수정된 부분 끝 --- ✨

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

    # ✨ [수정] 학습 기록 불러오기 경로 설정
    history_load_path = None
    if args.model_path:
        # 불러올 모델과 같은 디렉토리에서 학습 기록 파일을 찾습니다.
        model_dir = os.path.dirname(args.model_path)
        potential_history_path = os.path.join(model_dir, "training_history.json")
        if os.path.exists(potential_history_path):
            history_load_path = potential_history_path
            print(f"✅ 이전 학습 기록을 발견했습니다: {history_load_path}")
        else:
            print(f"⚠️ 이전 학습 기록 파일을 찾지 못했습니다 ({potential_history_path}). 새로운 그래프를 시작합니다.")

    # ✨ [수정] 콜백 생성 시, load_history_from 인자 전달
    enhanced_callback = EnhancedVisualCallback(
        eval_env=eval_record_env,
        best_model_save_path=model_path,
        eval_freq=args.video_interval,
        n_eval_episodes=3,
        show_duration_seconds=20,
        save_videos=True,
        load_history_from=history_load_path, # 이 부분을 추가했습니다.
    )
    
    video_callback = VideoRecordingCallback(
        record_env=eval_record_env,
        record_interval_timesteps=args.video_interval,
        show_duration_seconds=args.video_duration
    )
    
    from training_callback import CurriculumCallback
    curriculum_callback = CurriculumCallback(
        total_timesteps=args.total_timesteps,
        initial_rand_power=args.rand_power
    )

    callback_list = CallbackList([enhanced_callback, video_callback, curriculum_callback])

    # ✨ [수정] PPO 모델을 불러오거나 생성할 때 learning_rate 인자 추가
    if args.model_path is not None:
        model = PPO.load(
            path=args.model_path, 
            env=vec_env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=args.learning_rate  # 기존 모델에 새로운 학습률 적용
        )
        print(f"✅ 모델을 성공적으로 불러왔습니다: {args.model_path}")
    else:
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=1, 
            tensorboard_log=LOG_DIR,
            learning_rate=args.learning_rate  # 새 모델에 학습률 적용
        )

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
    parser.add_argument(
        "--video_duration",
        type=int,
        default=120,
        help="Duration of the video to record",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=300_000,
        help="Number of timesteps interval to record the video",
    )
    parser.add_argument(
        "--rand_power",
        type=float,
        default=0.0,
        help="Amount of randomization to apply to joint angles at reset. 0.0 means no randomization.",
    )
    
    # ✨ [신규 추가] learning_rate 인자
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,  # PPO의 기본값
        help="The learning rate for the PPO optimizer.",
    )

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)