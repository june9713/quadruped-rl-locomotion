#!/usr/bin/env python3
"""
최대 환경 개수 테스트 스크립트
"""

import psutil
import time
import gc
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from go1_mujoco_env import Go1MujocoEnv

def test_max_environments():
    """시스템에서 지원 가능한 최대 환경 개수 테스트"""
    
    print("🔍 시스템 리소스 확인...")
    print(f"CPU 코어: {psutil.cpu_count(logical=False)}개")
    print(f"CPU 스레드: {psutil.cpu_count(logical=True)}개")
    print(f"총 메모리: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"사용 가능 메모리: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 단계별로 환경 개수 증가
    test_counts = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48]
    max_stable_envs = 0
    
    for num_envs in test_counts:
        print(f"\n🧪 {num_envs}개 환경 테스트 중...")
        
        try:
            # 메모리 사용량 측정 시작
            initial_memory = psutil.virtual_memory().used / (1024**3)
            
            # 환경 생성
            start_time = time.time()
            vec_env = make_vec_env(
                Go1MujocoEnv,
                env_kwargs={"ctrl_type": "position"},
                n_envs=num_envs,
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
            
            creation_time = time.time() - start_time
            
            # 간단한 테스트 실행
            print(f"  ⏱️  환경 생성 시간: {creation_time:.2f}초")
            
            # 몇 스텝 실행해보기
            obs = vec_env.reset()
            for i in range(10):
                actions = [vec_env.action_space.sample() for _ in range(num_envs)]
                obs, rewards, dones, infos = vec_env.step(actions)
                
                # CPU 및 메모리 사용량 체크
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_used = psutil.virtual_memory().used / (1024**3)
                memory_percent = psutil.virtual_memory().percent
                
                if i == 5:  # 중간에 한 번 출력
                    print(f"  💻 CPU 사용률: {cpu_percent:.1f}%")
                    print(f"  🧠 메모리 사용량: {memory_used:.1f} GB ({memory_percent:.1f}%)")
                    print(f"  📊 환경당 메모리: {(memory_used - initial_memory) / num_envs * 1024:.0f} MB")
            
            # 환경 정리
            vec_env.close()
            del vec_env
            gc.collect()
            
            # 성공적으로 완료
            max_stable_envs = num_envs
            print(f"  ✅ {num_envs}개 환경 성공!")
            
            # 메모리 사용률이 80% 넘으면 경고
            if memory_percent > 80:
                print(f"  ⚠️  메모리 사용률 높음 ({memory_percent:.1f}%)")
                break
                
            # CPU 사용률이 95% 넘으면 경고  
            if cpu_percent > 95:
                print(f"  ⚠️  CPU 사용률 높음 ({cpu_percent:.1f}%)")
                break
                
        except Exception as e:
            print(f"  ❌ {num_envs}개 환경 실패: {str(e)}")
            break
        
        # 잠시 대기 (시스템 안정화)
        time.sleep(2)
    
    print(f"\n{'='*50}")
    print(f"🎯 테스트 결과:")
    print(f"최대 안정 환경 개수: {max_stable_envs}개")
    print(f"{'='*50}")
    
    # 권장 설정 제안
    conservative = max(4, max_stable_envs // 2)
    recommended = max(8, int(max_stable_envs * 0.7))
    aggressive = max(12, int(max_stable_envs * 0.9))
    
    print(f"\n💡 권장 설정:")
    print(f"  보수적 (안정성 우선): --num_envs {conservative}")
    print(f"  권장 (균형): --num_envs {recommended}")
    print(f"  적극적 (성능 우선): --num_envs {aggressive}")
    
    return max_stable_envs

def quick_benchmark(num_envs_list=[8, 16, 24]):
    """빠른 성능 벤치마크"""
    
    print("🚀 빠른 성능 벤치마크...")
    
    results = []
    
    for num_envs in num_envs_list:
        print(f"\n📊 {num_envs}개 환경 벤치마크...")
        
        try:
            # 환경 생성
            start_time = time.time()
            vec_env = make_vec_env(
                Go1MujocoEnv,
                env_kwargs={"ctrl_type": "position"},
                n_envs=num_envs,
                seed=42,
                vec_env_cls=SubprocVecEnv,
            )
            creation_time = time.time() - start_time
            
            # 100 스텝 실행
            obs = vec_env.reset()
            start_time = time.time()
            
            for i in range(100):
                actions = [vec_env.action_space.sample() for _ in range(num_envs)]
                obs, rewards, dones, infos = vec_env.step(actions)
            
            step_time = time.time() - start_time
            steps_per_second = (100 * num_envs) / step_time
            
            # 리소스 사용량
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory_percent = psutil.virtual_memory().percent
            
            results.append({
                'num_envs': num_envs,
                'creation_time': creation_time,
                'steps_per_second': steps_per_second,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            })
            
            print(f"  ⏱️  생성 시간: {creation_time:.2f}초")
            print(f"  🏃 Steps/초: {steps_per_second:.0f}")
            print(f"  💻 CPU: {cpu_percent:.1f}%")
            print(f"  🧠 메모리: {memory_percent:.1f}%")
            
            vec_env.close()
            del vec_env
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 실패: {str(e)}")
            break
    
    # 최적 설정 찾기
    if results:
        best = max(results, key=lambda x: x['steps_per_second'] / x['cpu_percent'])
        print(f"\n🏆 최적 설정: {best['num_envs']}개 환경")
        print(f"   성능/효율 비율이 가장 좋습니다!")

if __name__ == "__main__":
    print("Intel Ultra 9 275HX 환경 개수 테스트")
    print("="*50)
    
    # 전체 테스트 또는 빠른 벤치마크 선택
    choice = input("1: 전체 테스트, 2: 빠른 벤치마크 (1 or 2): ").strip()
    
    if choice == "1":
        test_max_environments()
    else:
        quick_benchmark()