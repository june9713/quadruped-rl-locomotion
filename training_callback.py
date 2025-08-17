#!/usr/bin/env python3
import threading
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from collections import deque, defaultdict
import os
import time
import imageio.v2 as imageio
import pandas as pd
import datetime
from go1_mujoco_env import Go1MujocoEnv
import copy
import subprocess
import sys
import json
from matplotlib.gridspec import GridSpec
import queue # <--- 이 라인을 추가해주세요.



class CurriculumCallback(BaseCallback):
    """
    학습 진행률에 따라 환경의 'rand_power'를 조절하는 커리큘럼 콜백입니다.

    이 콜백은 학습 시작 시 사용자가 지정한 'rand_power' 값에서 시작하여,
    전체 학습 timesteps의 70% 지점에 도달할 때까지 선형적으로 감소시켜 0으로 만듭니다.
    70% 이후부터는 'rand_power'를 0으로 유지하여 안정적인 정책을 미세 조정합니다.
    """
    def __init__(self, total_timesteps: int, initial_rand_power: float, verbose: int = 0):
        """
        CurriculumCallback 인스턴스를 초기화합니다.

        :param total_timesteps: 모델 학습에 사용될 총 타임스텝 수.
        :param initial_rand_power: 학습 시작 시 적용할 초기 랜덤화 강도.
        :param verbose: 상세 정보 출력 레벨.
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_rand_power = initial_rand_power
        # 커리큘럼이 종료되는 시점 (전체 학습의 70%)
        self.curriculum_end_step = int(total_timesteps * 0.7)
        self._last_logged_power = -1.0

    def _on_step(self) -> bool:
        """
        학습의 매 스텝마다 호출되어 rand_power를 동적으로 조절합니다.
        """
        current_step = self.num_timesteps

        if current_step < self.curriculum_end_step:
            # 70% 지점에 도달할 때까지 rand_power를 선형적으로 감소시킵니다.
            # 진행률 (0.0 ~ 1.0) 계산
            progress = current_step / self.curriculum_end_step
            new_rand_power = self.initial_rand_power * (1.0 - progress)
        else:
            # 70% 지점을 넘어서면 rand_power를 0으로 고정합니다.
            new_rand_power = 0.0

        # 모든 병렬 환경에 새로운 rand_power 값을 설정합니다.
        # Go1MujocoEnv 클래스 내부에서는 '_rand_power'라는 이름의 속성을 사용합니다.
        self.training_env.set_attr("_rand_power", new_rand_power)

        # rand_power 값의 변화를 TensorBoard에 로깅합니다.
        self.logger.record("curriculum/rand_power", new_rand_power)
        
        return True


def check_and_install_moviepy():
    """moviepy 설치 확인 및 자동 설치"""
    try:
        import moviepy
        print("✅ moviepy가 이미 설치되어 있습니다.")
        return True
    except ImportError:
        print("📦 moviepy가 설치되지 않았습니다. 자동으로 설치합니다...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
            print("✅ moviepy 설치 완료!")
            return True
        except subprocess.CalledProcessError:
            print("❌ moviepy 설치 실패. 수동으로 설치해주세요: pip install moviepy")
            return False


class VisualTrainingCallback(BaseCallback):
    """학습 중간에 시각적으로 성능을 보여주는 콜백"""
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 300_000,
        n_eval_episodes: int = 3,
        show_duration_seconds: int = 30,
        save_videos: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.show_duration_seconds = show_duration_seconds
        self.save_videos = save_videos
        self.step_zero = True
        
        self.last_eval_timestep = 0
        self.eval_count = 0
        self.performance_history = deque(maxlen=50)
        
        # 데이터 저장용
        self.rewards_history = []
        self.lengths_history = []
        self.success_rates = []
        self.timesteps_history = []
        
    def _on_step(self) -> bool:
        # eval_freq 간격으로 평가를 수행하며, 학습 시작 시점(self.step_zero)에도 첫 평가를 수행합니다.
        if self.step_zero or (self.num_timesteps - self.last_eval_timestep >= self.eval_freq):
            self.step_zero = False
            self._evaluate_and_visualize()
            self.last_eval_timestep = self.num_timesteps
            
        return True
    
    def _evaluate_and_visualize(self):
        """모델 평가 및 시각화 + MP4 저장"""
        self.eval_count += 1

        print(f"\n{'='*60}")
        print(f"📊 평가 #{self.eval_count} (Timestep: {self.num_timesteps:,})")
        print(f"⏰ 시뮬레이션 시간: {self.show_duration_seconds}초")
        print(f"{'='*60}")

        episode_rewards = []
        episode_lengths = []
        success_count = 0
        self.n_eval_episodes  =2
        for episode in range(self.n_eval_episodes):
            print(f"  🎮 에피소드 {episode + 1}/{self.n_eval_episodes} 시뮬레이션 중...")

            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            frames = []

            start_time = time.time()

            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                episode_reward += float(reward)
                episode_length += 1

                # 프레임 캡처
                frame = None
                try:
                    frame = self.eval_env.render()
                    if isinstance(frame, list):
                        frame = frame[0]
                except Exception:
                    try:
                        frame = self.eval_env.render(mode="rgb_array")
                    except Exception:
                        frame = None

                if frame is not None:
                    frames.append(frame)

                time.sleep(0.02)

                if time.time() - start_time >= self.show_duration_seconds:
                    print(f"    ⏰ 시간 제한 ({self.show_duration_seconds}초) 도달")
                    break

                if terminated or truncated:
                    if info.get('bipedal_success', False):  # <-- 수정된 코드
                        success_count += 1
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"    📈 보상: {episode_reward:.2f}, 길이: {episode_length}")

        # 결과 집계
        mean_r, std_r = np.mean(episode_rewards), np.std(episode_rewards)
        mean_l, std_l = np.mean(episode_lengths), np.std(episode_lengths)
        success_rate = success_count / self.n_eval_episodes

        self.rewards_history.append(mean_r)
        self.lengths_history.append(mean_l)
        self.success_rates.append(success_rate)
        self.timesteps_history.append(self.num_timesteps)

        print(f"\n📊 평가 결과:")
        print(f"  평균 보상: {mean_r:.2f} ± {std_r:.2f}")
        print(f"  평균 길이: {mean_l:.1f} ± {std_l:.1f}")
        print(f"  성공률: {success_rate:.1%} ({success_count}/{self.n_eval_episodes})")
        print(f"{'='*60}\n")

        self._update_plots()
   
    def _update_plots(self):
        pass
        
   
    def save_progress_report(self, save_path: str):
        """진행 상황 보고서 저장"""
        if not self.rewards_history:
            return
            
        # 상세 보고서 생성
        plt.figure(figsize=(15, 10))
        
        # 2x2 서브플롯
        plt.subplot(2, 2, 1)
        plt.plot(self.timesteps_history, self.rewards_history, 'b-o', linewidth=2)
        plt.title('학습 진행: 평균 보상', fontsize=14)
        plt.xlabel('Timesteps')
        plt.ylabel('평균 보상')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.timesteps_history, self.lengths_history, 'g-o', linewidth=2)
        plt.title('학습 진행: 평균 에피소드 길이', fontsize=14)
        plt.xlabel('Timesteps')
        plt.ylabel('평균 길이')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.timesteps_history, [r*100 for r in self.success_rates], 'r-o', linewidth=2)
        plt.title('학습 진행: 성공률', fontsize=14)
        plt.xlabel('Timesteps')
        plt.ylabel('성공률 (%)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        plt.subplot(2, 2, 4)
        if len(self.rewards_history) > 5:
            # 학습 곡선의 기울기 (개선 속도)
            improvement_rate = np.diff(self.rewards_history)
            plt.plot(self.timesteps_history[1:], improvement_rate, 'purple', linewidth=2)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('학습 개선 속도', fontsize=14)
            plt.xlabel('Timesteps')
            plt.ylabel('보상 개선량')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV로 데이터 저장
        df = pd.DataFrame({
            'timesteps': self.timesteps_history,
            'mean_reward': self.rewards_history,
            'mean_length': self.lengths_history,
            'success_rate': self.success_rates
        })
        df.to_csv(f"{save_path}/training_data.csv", index=False)
        
        print(f"📊 진행 상황 보고서 저장: {save_path}")


# 파일명: training_callback.py

class EnhancedVisualCallback(VisualTrainingCallback):
    """
    개선된 시각화 콜백 - 실시간 그래프를 이미지 파일로 저장합니다.
    """
    
    def __init__(self, *args, use_curriculum=False, best_model_save_path: str = None, load_history_from: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_curriculum = use_curriculum
        
        # 최고 모델 저장을 위한 설정
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # 추가 추적 데이터
        self.reward_components_history = []
        self.curriculum_stages = []
        self.stability_metrics = []
        self.failure_reasons = []
        self.explained_variance_history = []
        self.explained_variance_timesteps = []
        
        # --- 스레딩 관련 코드 제거 ---
        print("\n📈 실시간 학습 그래프는 현재 디렉토리에 'realtime_progress.png' 파일로 주기적으로 저장됩니다.")
        
        # ✨ [추가] 학습 기록 불러오기
        if load_history_from:
            self._load_history(load_history_from)

    def _load_history(self, path: str):
        """저장된 학습 기록을 .json 파일에서 불러옵니다."""
        if not os.path.exists(path):
            print(f"⚠️ 학습 기록 파일을 찾을 수 없습니다: {path}. 새로운 기록을 시작합니다.")
            return
        
        try:
            with open(path, 'r') as f:
                history = json.load(f)
            
            # 이전 기록 불러오기
            self.best_mean_reward = history.get('best_mean_reward', -np.inf)
            self.rewards_history = history.get('rewards_history', [])
            self.lengths_history = history.get('lengths_history', [])
            self.success_rates = history.get('success_rates', [])
            self.timesteps_history = history.get('timesteps_history', [])
            self.reward_components_history = history.get('reward_components_history', [])
            self.stability_metrics = history.get('stability_metrics', [])
            self.failure_reasons = history.get('failure_reasons', [])
            
            # eval_count는 기록된 데이터의 길이로 설정
            self.eval_count = len(self.rewards_history)
            
            if self.timesteps_history:
                print(f"✅ 학습 기록을 성공적으로 불러왔습니다: {path}")
                print(f"   - {self.eval_count}개의 이전 평가 지점에서 이어갑니다.")
                print(f"   - 마지막 Timestep: {self.timesteps_history[-1]:,}, 최고 보상: {self.best_mean_reward:.2f}")
            else:
                print("   - 불러온 기록이 비어있습니다. 새로운 기록을 시작합니다.")

        except Exception as e:
            print(f"❌ 학습 기록 파일 불러오기 오류: {e}. 새로운 기록을 시작합니다.")
   
    def _evaluate_and_visualize(self):
        """개선된 평가 및 시각화"""
        self.eval_count += 1
        
        print(f"\n{'='*70}")
        print(f"📊 고급 평가 #{self.eval_count} (Timestep: {self.num_timesteps:,})")
        print(f"{'='*70}")
        
        # 기본 평가 데이터
        episode_rewards = []
        episode_lengths = []
        episode_components = []
        episode_stability = []
        episode_failures = []
        success_count = 0
        self.n_eval_episodes = 2
        for episode in range(self.n_eval_episodes):
            print(f"\n🎮 에피소드 {episode + 1}/{self.n_eval_episodes}")
            
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            # 에피소드별 상세 추적
            reward_components = {}
            stability_metrics = []
            frames = []
            
            start_time = time.time()
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += float(reward)
                episode_length += 1
                
                # 보상 컴포넌트 추적
                if 'upright' in info:
                    for key in ['upright', 'height', 'feet', 'forward_vel', 
                               'lateral_vel', 'cop_stab', 'zmp_stab']:
                        if key in info:
                            if key not in reward_components:
                                reward_components[key] = []
                            reward_components[key].append(info.get(key, 0))
                
                # 안정성 메트릭
                if 'stab_ang' in info:
                    stability_metrics.append({
                        'angular_stability': info.get('stab_ang', 0),
                        'cop_stability': info.get('cop_stab', 0),
                        'zmp_stability': info.get('zmp_stab', 0)
                    })
                
                # 프레임 캡처
                if self.save_videos:
                    try:
                        frame = self.eval_env.render()
                        if frame is not None:
                            frames.append(frame)
                    except:
                        pass
                
                time.sleep(0.01)
                
                # 종료 조건
                if time.time() - start_time >= self.show_duration_seconds:
                    break
                    
                if terminated or truncated:
                    if info.get('bipedal_success', False):
                        success_count += 1
                    else:
                        failure_reason = self._analyze_failure(info, obs)
                        episode_failures.append(failure_reason)
                    break
            
            # 에피소드 데이터 저장
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            avg_components = {}
            for key, values in reward_components.items():
                if values:
                    avg_components[key] = np.mean(values)
            episode_components.append(avg_components)
            
            if stability_metrics:
                avg_stability = {
                    key: np.mean([m[key] for m in stability_metrics])
                    for key in stability_metrics[0].keys()
                }
                episode_stability.append(avg_stability)
            
            print(f"  📈 보상: {episode_reward:.2f}")
            print(f"  ⏱️ 길이: {episode_length}")
            print(f"  🎯 주요 컴포넌트: {', '.join([f'{k}:{v:.2f}' for k,v in avg_components.items()][:3])}")
            
            if self.save_videos and frames:
                self._save_video(frames, episode, episode_reward)
        
        # 전체 평가 결과 저장 및 최고 모델 업데이트
        self._update_history(episode_rewards, episode_lengths, 
                            episode_components, episode_stability, 
                            episode_failures, success_count)
        
        # 플롯 이미지 파일로 저장
        self._update_enhanced_plots()
        
        print(f"\n{'='*70}")
        print(f"📊 평가 요약:")
        print(f"  평균 보상: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  성공률: {success_count/self.n_eval_episodes:.1%}")
        if self.use_curriculum and hasattr(self.eval_env, 'standing_reward'):
            print(f"  커리큘럼 단계: {self.eval_env.standing_reward.curriculum_stage}")
        print(f"{'='*70}\n")
   
    def _analyze_failure(self, info, obs):
        """실패 원인 분석"""
        reasons = []
        
        if info.get('upright', 1) < 0.5:
            reasons.append('fall_forward' if obs[0] > 0 else 'fall_backward')
        if info.get('height', 1) < 0.3:
            reasons.append('too_low')
        if info.get('lateral_vel', 1) < 0.5:
            reasons.append('lateral_instability')
        if info.get('joint_limit', 0) < -0.5:
            reasons.append('joint_limit_violation')
        
        return reasons[0] if reasons else 'unknown'
   
    def _save_video(self, frames, episode, reward):
        """비디오 저장"""
        os.makedirs("eval_videos", exist_ok=True)
        filename = (f"eval_videos/enhanced_eval{self.eval_count}_ep{episode+1}_"
                   f"r{reward:.0f}_t{self.num_timesteps}.mp4")
        try:
            imageio.mimsave(filename, frames, fps=30)
        except:
            pass
   
    def _update_history(self, rewards, lengths, components, stability, failures, successes):
        """히스토리 업데이트 및 최고 성능 모델 저장"""
        mean_reward = np.mean(rewards)
        self.rewards_history.append(mean_reward)
        self.lengths_history.append(np.mean(lengths))
        self.success_rates.append(successes / self.n_eval_episodes)
        self.timesteps_history.append(self.num_timesteps)
        
        if self.best_model_save_path is not None:
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"\n🚀 새로운 최고 평균 보상 달성: {self.best_mean_reward:.2f} (Timestep: {self.num_timesteps:,})")
                
                # 최고 모델 저장
                save_path = os.path.join(self.best_model_save_path, "best_model.zip")
                self.model.save(save_path)
                print(f"💾 최고 모델 저장 완료: {save_path}")

                # ✨ [추가] 학습 기록 저장 로직
                history_data = {
                    'best_mean_reward': self.best_mean_reward,
                    'rewards_history': self.rewards_history,
                    'lengths_history': self.lengths_history,
                    'success_rates': self.success_rates,
                    'timesteps_history': self.timesteps_history,
                    'reward_components_history': self.reward_components_history,
                    'stability_metrics': self.stability_metrics,
                    'failure_reasons': self.failure_reasons,
                }
                history_save_path = os.path.join(self.best_model_save_path, "training_history.json")
                try:
                    with open(history_save_path, 'w') as f:
                        # numpy 타입을 python 기본 타입으로 변환하여 저장
                        json.dump(history_data, f, indent=4, default=float)
                    print(f"💾 학습 기록 저장 완료: {history_save_path}")
                except Exception as e:
                    print(f"❌ 학습 기록 저장 실패: {e}")


        avg_components = {}
        if components:
            keys = components[0].keys()
            for key in keys:
                values = [c.get(key, 0) for c in components]
                avg_components[key] = np.mean(values)
        self.reward_components_history.append(avg_components)
        
        if stability:
            avg_stability = {}
            keys = stability[0].keys()
            for key in keys:
                values = [s.get(key, 0) for s in stability]
                avg_stability[key] = np.mean(values)
            self.stability_metrics.append(avg_stability)
        
        failure_counts = {}
        for f in failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1
        self.failure_reasons.append(failure_counts)
        
        if self.use_curriculum and hasattr(self.eval_env, 'standing_reward'):
            self.curriculum_stages.append(self.eval_env.standing_reward.curriculum_stage)
   
    def _update_enhanced_plots(self):
        """실시간으로 학습 진행 상황 그래프를 이미지 파일로 저장합니다."""
        if len(self.rewards_history) < 2:
            return

        fig = None  # 예외 발생 시 fig 변수가 없을 수 있으므로 초기화
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            timesteps = self.timesteps_history
            rewards = self.rewards_history

            ax.plot(timesteps, rewards, 'b-', linewidth=2, label='평균 보상')

            # 이동 평균선 추가
            if len(rewards) >= 10:
                window = 10
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(timesteps[window-1:], ma, 'r--', linewidth=2, label=f'이동평균 ({window}-evals)')

            ax.set_title('실시간 학습 진행 상황: 평균 보상', fontsize=16)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('평균 보상')
            ax.grid(True, alpha=0.4)
            ax.legend()
            fig.tight_layout()

            save_path = "./training_progress.png"
            plt.savefig(save_path, dpi=100)
            
        except Exception as e:
            print(f"❌ 실시간 그래프 저장 중 오류 발생: {e}")
        finally:
            if fig is not None:
                plt.close(fig) # 리소스 누수를 방지하기 위해 항상 figure를 닫음

    def save_detailed_analysis(self, save_path: str):
        """상세 분석 보고서 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        if len(self.reward_components_history) > 10:
            components_df = pd.DataFrame(self.reward_components_history)
            
            plt.figure(figsize=(10, 8))
            corr = components_df.corr()
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title('보상 컴포넌트 간 상관관계')
            
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                           ha='center', va='center',
                           color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/component_correlation.png", dpi=300)
            plt.close()
        
        if self.use_curriculum and self.curriculum_stages:
            plt.figure(figsize=(12, 8))
            
            stage_success = {}
            for i, stage in enumerate(self.curriculum_stages):
                if stage not in stage_success:
                    stage_success[stage] = []
                stage_success[stage].append(self.success_rates[i])
            
            plt.subplot(2, 1, 1)
            for stage, rates in stage_success.items():
                plt.bar(stage, np.mean(rates), alpha=0.7, 
                       label=f'Stage {stage}')
            plt.xlabel('커리큘럼 단계')
            plt.ylabel('평균 성공률')
            plt.title('커리큘큘럼 단계별 성공률')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(self.timesteps_history, self.curriculum_stages, 'o-')
            plt.xlabel('Timesteps')
            plt.ylabel('커리큘럼 단계')
            plt.title('커리큘럼 진행 추이')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/curriculum_analysis.png", dpi=300)
            plt.close()
        
        analysis_data = {
            'summary': {
                'total_evaluations': len(self.rewards_history),
                'final_reward': self.rewards_history[-1] if self.rewards_history else 0,
                'final_success_rate': self.success_rates[-1] if self.success_rates else 0,
                'best_reward': max(self.rewards_history) if self.rewards_history else 0,
                'best_success_rate': max(self.success_rates) if self.success_rates else 0,
            },
            'history': {
                'timesteps': self.timesteps_history,
                'rewards': self.rewards_history,
                'success_rates': self.success_rates,
                'episode_lengths': self.lengths_history,
            }
        }
        
        with open(f"{save_path}/analysis_data.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"📊 상세 분석 보고서 저장 완료: {save_path}")


class VideoRecordingCallback(BaseCallback):
    def __init__(
        self,
        record_env,
        record_interval_timesteps: int = 100_000,
        record_episodes: int = 1,
        video_folder: str = "training_videos",
        show_duration_seconds: int = 15,
    ):
        super().__init__()
        self.record_env = record_env
        self.record_interval = record_interval_timesteps
        self.record_episodes = record_episodes
        self.video_folder = video_folder
        self.show_duration_seconds = show_duration_seconds
        self.last_record_timestep = 0
        
        os.makedirs(video_folder, exist_ok=True)
        
        # moviepy 설치 확인
        self.moviepy_available = check_and_install_moviepy()
        if not self.moviepy_available:
            print("⚠️ 비디오 저장이 비활성화됩니다.")
        
        print(f"🎥 비디오 녹화 설정: {show_duration_seconds}초간 녹화")
    
    # 파일명: training_callback.py -> 클래스명: VideoRecordingCallback

    def _on_step(self) -> bool:
        if not self.moviepy_available:
            return True
            
        if self.num_timesteps - self.last_record_timestep >= self.record_interval:
            self._record_video()
            self.last_record_timestep = self.num_timesteps
        return True
    
    def _record_video(self):
        """원하는 길이의 비디오를 정확히 녹화하도록 수정한 함수"""
        print(f"\n🎥 비디오 녹화 중... (Timestep: {self.num_timesteps:,})")
        
        termination_counts = defaultdict(int)
        total_terminations_in_video = 0
        
        try:
            obs, _ = self.record_env.reset()
            frames = []
            episode_reward = 0

            # 목표 영상 길이와 FPS 설정
            target_video_seconds = self.show_duration_seconds
            fps = 30
            num_frames_to_record = target_video_seconds * fps
            
            # 시간 기반 루프를 프레임 수 기반 루프로 변경
            while len(frames) < num_frames_to_record:
                
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.record_env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                frame = self.record_env.render()
                if isinstance(frame, list):
                    frame = frame[0]
                frames.append(frame)
                
                is_done = terminated or truncated
                if is_done:
                    current_info = info[0] if isinstance(info, list) else info
                    reason = current_info.get('termination_reason')
                    
                    if reason and reason != 'not_terminated':
                        base_reason = reason.split(' (')[0]
                        termination_counts[base_reason] += 1
                        total_terminations_in_video += 1

                        # ✨ 추가된 부분: 상세한 종료 원인과 값을 직접 출력
                        details = current_info.get('termination_details', '세부 정보 없음.')
                        #print(f"      - 종료 발생: {base_reason}")
                        #print(f"        └> 상세 정보: {details}")
                    
                    obs, _ = self.record_env.reset()
            
            if frames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # 파일명에서 보상 값을 소수점 없이 정수로 표시하도록 수정
                filename = f"{self.video_folder}/training_t{self.num_timesteps}_r{int(episode_reward)}_{timestamp}.mp4"
                imageio.mimsave(filename, frames, fps=fps) # 설정한 fps 값 사용
                print(f"✅ 비디오 저장: {filename} (보상: {episode_reward:.1f})")

            if total_terminations_in_video > 0:
                print("-----------------------------------------")
                print("| Termination Reasons (During Video)    |")
                print(f"| Total terminations: {total_terminations_in_video:<16} |")
                print("-----------------------------------------")
                
                sorted_reasons = sorted(termination_counts.items(), key=lambda item: item[1], reverse=True)

                for reason, count in sorted_reasons:
                    percentage = (count / total_terminations_in_video) * 100
                    print(f"| {reason:<25} | {count:<5} ({percentage:>5.1f}%) |")
                print("-----------------------------------------")

        except Exception as e:
            import traceback
            print(f"❌ 비디오 녹화 실패: {str(e)}")
            traceback.print_exc()