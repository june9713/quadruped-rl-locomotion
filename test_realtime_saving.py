#!/usr/bin/env python3
"""
실시간 저장 기능 테스트 스크립트
기존 기능은 전혀 건드리지 않고, 새로운 실시간 저장 기능만 테스트합니다.
"""

import os
import sys
import time
import json
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from training_callback import RealTimeSavingCallback, ComprehensiveSavingCallback
from go1_mujoco_env import Go1MujocoEnv

def test_realtime_saving():
    """실시간 저장 기능을 테스트합니다."""
    print("🧪 실시간 저장 기능 테스트 시작")
    
    # 테스트용 저장 디렉토리
    test_save_dir = "test_realtime_data"
    
    # 환경 생성
    print("📦 테스트 환경 생성 중...")
    env = Go1MujocoEnv(
        ctrl_type="position",
        biped=False,
        rand_power=0.1,
        action_noise=0.05
    )
    
    # 실시간 저장 콜백 생성
    print("🔧 실시간 저장 콜백 생성 중...")
    saving_callback = RealTimeSavingCallback(
        save_dir=test_save_dir,
        save_frequency=10,  # 테스트를 위해 빠른 저장 주기
        checkpoint_frequency=50,  # 테스트를 위해 빠른 체크포인트 주기
        verbose=1
    )
    
    # 콜백 초기화
    print("🚀 콜백 초기화 중...")
    saving_callback._on_training_start()
    
    # 시뮬레이션 스텝 실행 (테스트용)
    print("🎮 시뮬레이션 스텝 실행 중...")
    for step in range(100):  # 100 스텝 테스트
        # 랜덤 액션 생성
        action = env.action_space.sample()
        
        # 환경 스텝 실행
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 콜백 스텝 실행
        saving_callback.num_timesteps = step
        saving_callback._on_step()
        
        if step % 20 == 0:
            print(f"  📊 스텝 {step}: 보상={reward:.3f}, 종료={terminated}, 잘림={truncated}")
        
        # 에피소드 종료 시 리셋
        if terminated or truncated:
            obs = env.reset()
            print(f"  🔄 에피소드 리셋 (스텝 {step})")
        
        time.sleep(0.01)  # 빠른 테스트를 위한 짧은 대기
    
    # 학습 종료 처리
    print("🔚 학습 종료 처리 중...")
    saving_callback.on_training_end()
    
    # 저장된 파일들 확인
    print("\n📁 저장된 파일들 확인:")
    if os.path.exists(test_save_dir):
        for root, dirs, files in os.walk(test_save_dir):
            level = root.replace(test_save_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("  ❌ 저장 디렉토리를 찾을 수 없습니다.")
    
    # 샘플 데이터 확인
    print("\n📊 샘플 데이터 확인:")
    try:
        # 에피소드 데이터 확인
        episodes_dir = f"{test_save_dir}/episodes"
        if os.path.exists(episodes_dir):
            episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]
            if episode_files:
                latest_file = sorted(episode_files)[-1]
                with open(f"{episodes_dir}/{latest_file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  📈 최신 에피소드 데이터: {len(data)}개 항목")
                if data:
                    print(f"    - 첫 번째 항목: {list(data[0].keys())}")
        
        # 환경 상태 데이터 확인
        env_states_dir = f"{test_save_dir}/environment_states"
        if os.path.exists(env_states_dir):
            env_files = [f for f in os.listdir(env_states_dir) if f.endswith('.json')]
            if env_files:
                latest_file = sorted(env_files)[-1]
                with open(f"{env_states_dir}/{latest_file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  🌍 최신 환경 상태 데이터: {len(data)}개 항목")
                if data:
                    print(f"    - 첫 번째 항목: {list(data[0].keys())}")
        
        # 하이퍼파라미터 확인
        hyper_dir = f"{test_save_dir}/hyperparameters"
        if os.path.exists(hyper_dir):
            hyper_files = [f for f in os.listdir(hyper_dir) if f.endswith('.json')]
            if hyper_files:
                latest_file = sorted(hyper_files)[-1]
                with open(f"{hyper_dir}/{latest_file}", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"  ⚙️ 하이퍼파라미터: {len(data)}개 항목")
                print(f"    - 모델 타입: {data.get('model_type', 'N/A')}")
                print(f"    - 정책 타입: {data.get('policy_type', 'N/A')}")
        
    except Exception as e:
        print(f"  ❌ 데이터 확인 중 오류: {e}")
    
    print("\n✅ 실시간 저장 기능 테스트 완료!")
    print(f"📁 테스트 데이터 위치: {os.path.abspath(test_save_dir)}")

if __name__ == "__main__":
    test_realtime_saving()
