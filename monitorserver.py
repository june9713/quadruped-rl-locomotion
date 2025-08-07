import threading
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  # PPO 모델을 사용하기 위해 임포트
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym
from collections import deque
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
        eval_interval_minutes: int = 10,
        n_eval_episodes: int = 3,
        show_duration_seconds: int = 30,
        save_videos: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval_seconds = eval_interval_minutes * 60
        self.n_eval_episodes = n_eval_episodes
        self.show_duration_seconds = show_duration_seconds
        self.save_videos = save_videos
        self.step_zero = True
        
        self.last_eval_time = time.time()
        self.eval_count = 0
        self.performance_history = deque(maxlen=50)
        
        # 실시간 플롯을 위한 설정
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('실시간 학습 진행 상황', fontsize=16)
        
        # 서브플롯 설정
        self.reward_ax = self.axes[0, 0]
        self.episode_length_ax = self.axes[0, 1] 
        self.success_rate_ax = self.axes[1, 0]
        self.learning_curve_ax = self.axes[1, 1]
        
        # 데이터 저장용
        self.rewards_history = []
        self.lengths_history = []
        self.success_rates = []
        self.timesteps_history = []
        
    def _on_step(self) -> bool:
        current_time = time.time()
        
        # 지정된 시간 간격마다 평가 및 시각화
        if self.step_zero or (current_time - self.last_eval_time >= self.eval_interval_seconds):
            self.step_zero = False
            self._evaluate_and_visualize()
            self.last_eval_time = current_time
            
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
                    if info.get('standing_success', False):
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
        """실시간 플롯 업데이트"""
        
        # 1. 보상 추이
        self.reward_ax.clear()
        if self.rewards_history:
            self.reward_ax.plot(self.timesteps_history, self.rewards_history, 'b-o', linewidth=2)
            self.reward_ax.set_title('평균 보상 추이')
            self.reward_ax.set_xlabel('Timesteps')
            self.reward_ax.set_ylabel('평균 보상')
            self.reward_ax.grid(True, alpha=0.3)
        
        # 2. 에피소드 길이 추이
        self.episode_length_ax.clear()
        if self.lengths_history:
            self.episode_length_ax.plot(self.timesteps_history, self.lengths_history, 'g-o', linewidth=2)
            self.episode_length_ax.set_title('평균 에피소드 길이')
            self.episode_length_ax.set_xlabel('Timesteps')
            self.episode_length_ax.set_ylabel('평균 길이')
            self.episode_length_ax.grid(True, alpha=0.3)
        
        # 3. 성공률 추이
        self.success_rate_ax.clear()
        if self.success_rates:
            self.success_rate_ax.plot(self.timesteps_history, 
                                      [r*100 for r in self.success_rates], 'r-o', linewidth=2)
            self.success_rate_ax.set_title('성공률 추이')
            self.success_rate_ax.set_xlabel('Timesteps')
            self.success_rate_ax.set_ylabel('성공률 (%)')
            self.success_rate_ax.grid(True, alpha=0.3)
            self.success_rate_ax.set_ylim(0, 100)
        
        # 4. 최근 성능 (박스플롯 또는 히스토그램)
        self.learning_curve_ax.clear()
        if len(self.rewards_history) > 1:
            recent_rewards = self.rewards_history[-10:]
            self.learning_curve_ax.hist(recent_rewards, bins=max(3, len(recent_rewards)//2), 
                                        alpha=0.7, color='purple')
            self.learning_curve_ax.set_title('최근 보상 분포')
            self.learning_curve_ax.set_xlabel('보상')
            self.learning_curve_ax.set_ylabel('빈도')
            self.learning_curve_ax.axvline(np.mean(recent_rewards), color='red', 
                                           linestyle='--', label=f'평균: {np.mean(recent_rewards):.2f}')
            self.learning_curve_ax.legend()
        
        plt.tight_layout()
        plt.pause(0.1)
   
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


class EnhancedVisualCallback(VisualTrainingCallback):
    """개선된 시각화 콜백 - 더 많은 분석 기능"""
    
    def __init__(self, *args, use_curriculum=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_curriculum = use_curriculum
        
        # 추가 추적 데이터
        self.reward_components_history = []
        self.curriculum_stages = []
        self.stability_metrics = []
        self.failure_reasons = []
        
        # --- 레이아웃 수정 ---
        # 개선된 플롯 설정 - 평균 보상 그래프를 최상단에 배치
        plt.ioff()
        self.fig = plt.figure(figsize=(16, 18))  # 세로 길이를 늘려 그래프 공간 확보
        gs = GridSpec(4, 2, figure=self.fig, hspace=0.4, wspace=0.3)

        # 행 0: 평균 보상 (전체 너비)
        self.reward_ax = self.fig.add_subplot(gs[0, :])

        # 행 1: 보상 컴포넌트와 성공률
        self.components_ax = self.fig.add_subplot(gs[1, 0])
        self.success_ax = self.fig.add_subplot(gs[1, 1])

        # 행 2: 안정성 메트릭과 실패 원인
        self.stability_ax = self.fig.add_subplot(gs[2, 0])
        self.failure_ax = self.fig.add_subplot(gs[2, 1])

        # 행 3: 보상 컴포넌트 히트맵 (전체 너비)
        self.heatmap_ax = self.fig.add_subplot(gs[3, :])
        
        # 원래 있던 두 개의 축은 더 이상 사용하지 않음
        self.learning_curve_ax = None
        self.episode_length_ax = None
        
        plt.ion()
   
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
                    # 성공/실패 분석
                    if info.get('standing_success', False):
                        success_count += 1
                    else:
                        # 실패 원인 분석
                        failure_reason = self._analyze_failure(info, obs)
                        episode_failures.append(failure_reason)
                    break
            
            # 에피소드 데이터 저장
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 평균 컴포넌트 값
            avg_components = {}
            for key, values in reward_components.items():
                if values:
                    avg_components[key] = np.mean(values)
            episode_components.append(avg_components)
            
            # 평균 안정성
            if stability_metrics:
                avg_stability = {
                    key: np.mean([m[key] for m in stability_metrics])
                    for key in stability_metrics[0].keys()
                }
                episode_stability.append(avg_stability)
            
            print(f"  📈 보상: {episode_reward:.2f}")
            print(f"  ⏱️ 길이: {episode_length}")
            print(f"  🎯 주요 컴포넌트: {', '.join([f'{k}:{v:.2f}' for k,v in avg_components.items()][:3])}")
            
            # 비디오 저장
            if self.save_videos and frames:
                self._save_video(frames, episode, episode_reward)
        
        # 전체 평가 결과 저장
        self._update_history(episode_rewards, episode_lengths, 
                            episode_components, episode_stability, 
                            episode_failures, success_count)
        
        # 플롯 업데이트
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
        """히스토리 업데이트"""
        self.rewards_history.append(np.mean(rewards))
        self.lengths_history.append(np.mean(lengths))
        self.success_rates.append(successes / self.n_eval_episodes)
        self.timesteps_history.append(self.num_timesteps)
        
        # 평균 컴포넌트
        avg_components = {}
        if components:
            keys = components[0].keys()
            for key in keys:
                values = [c.get(key, 0) for c in components]
                avg_components[key] = np.mean(values)
        self.reward_components_history.append(avg_components)
        
        # 안정성 메트릭
        if stability:
            avg_stability = {}
            keys = stability[0].keys()
            for key in keys:
                values = [s.get(key, 0) for s in stability]
                avg_stability[key] = np.mean(values)
            self.stability_metrics.append(avg_stability)
        
        # 실패 분석
        failure_counts = {}
        for f in failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1
        self.failure_reasons.append(failure_counts)
        
        # 커리큘럼 단계
        if self.use_curriculum and hasattr(self.eval_env, 'standing_reward'):
            self.curriculum_stages.append(self.eval_env.standing_reward.curriculum_stage)
   
    def _update_enhanced_plots(self):
        """개선된 플롯 업데이트"""
        plt.figure(self.fig.number)
        
        # 1. 전체 보상 추이
        self.reward_ax.clear()
        self.reward_ax.plot(self.timesteps_history, self.rewards_history, 'b-', linewidth=2)
        if len(self.rewards_history) > 10:
            # 이동 평균
            window = min(10, len(self.rewards_history))
            ma = np.convolve(self.rewards_history, np.ones(window)/window, mode='valid')
            ma_x = self.timesteps_history[window-1:]
            self.reward_ax.plot(ma_x, ma, 'r--', linewidth=2, label='이동평균')
        self.reward_ax.set_title('학습 진행: 평균 보상', fontsize=14, weight='bold') # 제목 강조
        self.reward_ax.set_xlabel('Timesteps')
        self.reward_ax.set_ylabel('평균 보상')
        self.reward_ax.grid(True, alpha=0.3)
        self.reward_ax.legend()
        
        # 2. 보상 컴포넌트 분석
        self.components_ax.clear()
        if self.reward_components_history:
            components_df = pd.DataFrame(self.reward_components_history)
            for col in components_df.columns[:5]:  # 상위 5개만
                self.components_ax.plot(self.timesteps_history, 
                                      components_df[col], 
                                      label=col, linewidth=1.5)
            self.components_ax.set_title('보상 컴포넌트 추이', fontsize=12)
            self.components_ax.set_xlabel('Timesteps')
            self.components_ax.set_ylabel('컴포넌트 값')
            self.components_ax.legend(loc='best', fontsize=8)
            self.components_ax.grid(True, alpha=0.3)
        
        # 3. 성공률
        self.success_ax.clear()
        self.success_ax.plot(self.timesteps_history, 
                           [r*100 for r in self.success_rates], 
                           'g-o', linewidth=2, markersize=6)
        if self.use_curriculum and self.curriculum_stages:
            # 커리큘럼 단계 표시
            ax2 = self.success_ax.twinx()
            ax2.plot(self.timesteps_history, self.curriculum_stages, 
                    'orange', linestyle='--', linewidth=1)
            ax2.set_ylabel('커리큘럼 단계', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        self.success_ax.set_title('성공률 추이', fontsize=12)
        self.success_ax.set_xlabel('Timesteps')
        self.success_ax.set_ylabel('성공률 (%)')
        self.success_ax.set_ylim(0, 105)
        self.success_ax.grid(True, alpha=0.3)
        
        # 4. 안정성 메트릭
        self.stability_ax.clear()
        if self.stability_metrics:
            stability_df = pd.DataFrame(self.stability_metrics)
            x = self.timesteps_history[:len(stability_df)]
            for col in stability_df.columns:
                self.stability_ax.plot(x, stability_df[col], 
                                     label=col.replace('_', ' '), linewidth=1.5)
            self.stability_ax.set_title('안정성 메트릭', fontsize=12)
            self.stability_ax.set_xlabel('Timesteps')
            self.stability_ax.set_ylabel('안정성 점수')
            self.stability_ax.legend(loc='best', fontsize=8)
            self.stability_ax.grid(True, alpha=0.3)
            self.stability_ax.set_ylim(0, 1.1)
        
        # 5. 보상 컴포넌트 히트맵
        self.heatmap_ax.clear()
        if len(self.reward_components_history) > 5:
            # 최근 데이터로 히트맵
            recent_components = pd.DataFrame(self.reward_components_history[-20:])
            if not recent_components.empty:
                data = recent_components.T.values
                im = self.heatmap_ax.imshow(data, aspect='auto', cmap='coolwarm')
                self.heatmap_ax.set_yticks(range(len(recent_components.columns)))
                self.heatmap_ax.set_yticklabels(recent_components.columns, fontsize=8)
                self.heatmap_ax.set_xlabel('최근 평가 (과거 → 현재)')
                self.heatmap_ax.set_title('보상 컴포넌트 히트맵', fontsize=12)
                plt.colorbar(im, ax=self.heatmap_ax, fraction=0.046, pad=0.04)
        
        # 6. 실패 원인 분석
        self.failure_ax.clear()
        if self.failure_reasons:
            # 전체 실패 원인 집계
            all_failures = {}
            for failure_dict in self.failure_reasons:
                for reason, count in failure_dict.items():
                    all_failures[reason] = all_failures.get(reason, 0) + count
            
            if all_failures:
                reasons = list(all_failures.keys())
                counts = list(all_failures.values())
                colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(reasons)))
                self.failure_ax.pie(counts, labels=reasons, colors=colors, 
                                  autopct='%1.0f%%', startangle=90)
                self.failure_ax.set_title('실패 원인 분포', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitle 공간 확보
        self.fig.suptitle('향상된 학습 진행 상황 리포트', fontsize=16, weight='bold')
        plt.pause(0.1)
   
    def save_detailed_analysis(self, save_path: str):
        """상세 분석 보고서 저장"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 컴포넌트 상관관계 분석
        if len(self.reward_components_history) > 10:
            components_df = pd.DataFrame(self.reward_components_history)
            
            plt.figure(figsize=(10, 8))
            corr = components_df.corr()
            plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title('보상 컴포넌트 간 상관관계')
            
            # 상관계수 표시
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                           ha='center', va='center',
                           color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/component_correlation.png", dpi=300)
            plt.close()
        
        # 2. 학습 단계별 분석
        if self.use_curriculum and self.curriculum_stages:
            plt.figure(figsize=(12, 8))
            
            # 스테이지별 성공률
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
            plt.title('커리큘럼 단계별 성공률')
            plt.legend()
            
            # 스테이지 진행 시간
            plt.subplot(2, 1, 2)
            plt.plot(self.timesteps_history, self.curriculum_stages, 'o-')
            plt.xlabel('Timesteps')
            plt.ylabel('커리큘럼 단계')
            plt.title('커리큘럼 진행 추이')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/curriculum_analysis.png", dpi=300)
            plt.close()
        
        # 3. JSON 형식으로 전체 데이터 저장
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
    
    def _on_step(self) -> bool:
        if not self.moviepy_available:
            return True
            
        if self.num_timesteps - self.last_record_timestep >= self.record_interval:
            self._record_video()
            self.last_record_timestep = self.num_timesteps
        return True
    
    def _record_video(self):
        """개선된 비디오 녹화"""
        print(f"\n🎥 비디오 녹화 중... (Timestep: {self.num_timesteps:,})")
        
        try:
            # 환경 리셋
            obs = self.record_env.reset()
            frames = []
            episode_reward = 0
            start_time = time.time()
            
            while time.time() - start_time < self.show_duration_seconds:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.record_env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                
                # 프레임 캡처
                frame = self.record_env.render(mode='rgb_array')
                if isinstance(frame, list):
                    frame = frame[0]
                frames.append(frame)
                
                if done[0] if isinstance(done, np.ndarray) else done:
                    obs = self.record_env.reset()
            
            # 비디오 저장
            if frames:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.video_folder}/training_t{self.num_timesteps}_r{episode_reward:.0f}_{timestamp}.mp4"
                imageio.mimsave(filename, frames, fps=30)
                print(f"✅ 비디오 저장: {filename} (보상: {episode_reward:.1f})")
                
        except Exception as e:
            print(f"❌ 비디오 녹화 실패: {str(e)}")

# ======================================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 메인 실행 블록 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ======================================================================================
if __name__ == "__main__":
    
    # 1. 학습에 사용할 로봇 환경 생성
    # Go1MujocoEnv의 설정은 필요에 따라 변경할 수 있습니다.
    env = Go1MujocoEnv()

    # 2. 모니터링 콜백 인스턴스 생성 (가장 기능이 많은 EnhancedVisualCallback 사용)
    # eval_interval_minutes: 평가 실행 간격 (분 단위)
    # show_duration_seconds: 평가 시 한 에피소드당 시뮬레이션 시간 (초)
    visual_callback = EnhancedVisualCallback(
        eval_env=env, 
        eval_interval_minutes=10,  # 10분마다 평가
        show_duration_seconds=20   # 20초 동안 시뮬레이션
    )

    # 3. 강화학습 모델 생성 (PPO 알고리즘)
    # policy="MlpPolicy": 다층 퍼셉트론(신경망) 정책 사용
    # verbose=1: 학습 진행 상황을 터미널에 출력
    model = PPO("MlpPolicy", env, verbose=1)

    print("====================== 학습 시작 ======================")
    
    try:
        # 4. 모델 학습 시작
        # total_timesteps: 총 학습할 횟수(타임스텝)
        # callback: 학습 중간에 호출할 콜백 지정
        model.learn(
            total_timesteps=2_000_000,
            callback=visual_callback
        )
    finally:
        # 5. 학습이 중단되거나 완료되면, 최종 분석 리포트 저장
        print("\n====================== 학습 종료 ======================")
        save_directory = "./training_reports"
        visual_callback.save_detailed_analysis(save_path=save_directory)
        print(f"최종 리포트가 '{save_directory}' 폴더에 저장되었습니다.")