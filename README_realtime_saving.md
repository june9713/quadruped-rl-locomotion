# 🚀 실시간 저장 기능 (Real-time Saving System)

## 📋 개요

이 프로젝트에 **기존 기능은 전혀 건드리지 않고** 새로운 실시간 저장 기능을 추가했습니다. 모든 기존 저장 시스템과 완벽히 호환되며, 학습 중단/재개를 위한 체크포인트와 상세한 학습 메트릭을 실시간으로 저장합니다.

## ✨ 새로 추가된 기능

### 1. **환경 상태 실시간 저장**
- 에피소드별 상세 정보 (보상, 길이, 성공률 등)
- 로봇의 현재 상태 (위치, 방향, 속도, 관절 정보)
- 환경 설정 및 하이퍼파라미터
- 발 접촉 정보 및 안정성 메트릭

### 2. **학습 메트릭 실시간 저장**
- 보상 및 비용 컴포넌트
- 학습 진행률 및 커리큘럼 정보
- 모델 상태 및 옵티마이저 정보
- 메모리 사용량 및 성능 지표

### 3. **환경 설정 및 하이퍼파라미터 저장**
- 학습 시작 시 모든 설정값 자동 저장
- 실험 재현성을 위한 환경 정보 보존
- 모델 아키텍처 및 학습 파라미터 기록

### 4. **학습 중단/재개를 위한 체크포인트**
- 정기적인 체크포인트 저장
- 학습 재개를 위한 상세 정보 제공
- 복구 지침 및 호환성 정보

## 🔧 사용법

### 기본 사용법

```python
from training_callback import ComprehensiveSavingCallback

# 실시간 저장 콜백 생성
realtime_saving_callback = ComprehensiveSavingCallback(
    save_dir="models/my_experiment/realtime_data",
    save_frequency=1000,        # 매 1000 스텝마다 저장
    checkpoint_frequency=10000, # 매 10000 스텝마다 체크포인트
    total_timesteps=5000000,    # 전체 학습 타임스텝
    verbose=1
)

# 기존 콜백들과 함께 사용
callback_list = CallbackList([
    enhanced_callback,
    video_callback, 
    curriculum_callback,
    realtime_saving_callback  # 새로 추가된 콜백
])
```

### 고급 설정

```python
# 개별 실시간 저장 콜백 사용
from training_callback import RealTimeSavingCallback

realtime_callback = RealTimeSavingCallback(
    save_dir="custom_data",
    save_frequency=500,           # 더 자주 저장
    save_episode_data=True,       # 에피소드 데이터 저장
    save_environment_state=True,   # 환경 상태 저장
    save_hyperparameters=True,    # 하이퍼파라미터 저장
    save_checkpoints=True,        # 체크포인트 저장
    checkpoint_frequency=5000,    # 체크포인트 주기
    verbose=1
)
```

## 📁 저장되는 데이터 구조

```
models/experiment_name/realtime_data/
├── episodes/                    # 에피소드별 상세 데이터
│   ├── episodes_20250101_120000.json
│   └── episodes_20250101_120100.json
├── environment_states/          # 환경 상태 정보
│   ├── env_states_20250101_120000.json
│   └── env_states_20250101_120100.json
├── learning_metrics/            # 학습 메트릭
│   ├── learning_metrics_20250101_120000.json
│   └── learning_metrics_20250101_120100.json
├── checkpoints/                 # 체크포인트
│   ├── checkpoint_20250101_120000.json
│   └── checkpoint_20250101_120100.json
└── hyperparameters/             # 하이퍼파라미터
    └── hyperparameters_20250101_120000.json
```

## 📊 저장되는 데이터 예시

### 에피소드 데이터
```json
{
  "timestep": 1000,
  "env_id": 0,
  "episode_count": 15,
  "success_count": 8,
  "success_rate": 0.533,
  "detailed_info": {
    "episode_info": {
      "current_step": 45,
      "time_remaining": 15.5
    },
    "robot_state": {
      "position": {"x": 0.25, "y": 0.0, "z": 0.6},
      "velocity": {"linear": [0.2, 0.0, 0.0]}
    },
    "performance_metrics": {
      "episode_progress": 0.75,
      "stability_score": 0.85
    }
  }
}
```

### 환경 상태 데이터
```json
{
  "timestep": 1000,
  "num_envs": 12,
  "rand_power": [0.1, 0.1, 0.1, ...],
  "environment_summary": {
    "environment_class": "Go1MujocoEnv",
    "ctrl_type": "position",
    "biped": false,
    "frame_skip": 10,
    "dt": 0.02
  }
}
```

### 학습 메트릭
```json
{
  "timestep": 1000,
  "total_timesteps": 1000,
  "training_progress": 0.0002,
  "recent_success_rate_avg": 0.55,
  "curriculum_progress": 0.3,
  "memory_usage_mb": 245.6
}
```

## 🔄 학습 재개 (Resume Training)

### 체크포인트에서 재개
```bash
python train.py --run train \
    --model_path models/experiment_name/checkpoints/checkpoint_20250101_120000.zip \
    --total_timesteps 5000000 \
    --run_name "resumed_experiment"
```

### 실시간 데이터 확인
```python
import json

# 체크포인트 정보 확인
with open("models/experiment_name/realtime_data/checkpoints/checkpoint_20250101_120000.json", "r") as f:
    checkpoint = json.load(f)

print(f"체크포인트 타임스텝: {checkpoint['timestep']}")
print(f"복구 가능: {checkpoint['recovery_info']['can_resume']}")
print(f"복구 지침: {checkpoint['recovery_info']['resume_instructions']}")
```

## 🧪 테스트

새로 추가된 실시간 저장 기능을 테스트하려면:

```bash
python test_realtime_saving.py
```

이 스크립트는:
- 테스트 환경에서 시뮬레이션 실행
- 실시간 저장 콜백 테스트
- 저장된 데이터 구조 확인
- 샘플 데이터 내용 검증

## ⚠️ 주의사항

### 1. **기존 기능 호환성**
- ✅ 기존 모든 저장 시스템과 완벽 호환
- ✅ 기존 콜백들과 함께 사용 가능
- ✅ 기존 모델 파일 형식 유지

### 2. **성능 고려사항**
- 저장 주기를 너무 짧게 설정하면 I/O 오버헤드 발생 가능
- 메모리 사용량을 고려하여 데이터 보관 기간 조절
- 대용량 데이터의 경우 주기적 정리 권장

### 3. **저장 공간**
- 실시간 데이터는 JSON 형식으로 저장되어 텍스트 기반
- 압축을 고려하여 주기적 아카이빙 권장
- 중요한 실험의 경우 백업 디렉토리 설정 권장

## 🚀 향후 개선 계획

### 1. **데이터 압축 및 최적화**
- JSON 대신 더 효율적인 바이너리 형식 지원
- 자동 압축 및 아카이빙 기능

### 2. **실시간 모니터링 대시보드**
- 웹 기반 실시간 학습 진행 상황 모니터링
- 그래프 및 차트를 통한 시각화

### 3. **분산 학습 지원**
- 멀티 머신 환경에서의 실시간 데이터 동기화
- 클라우드 스토리지 연동

## 📞 지원 및 문의

실시간 저장 기능과 관련된 질문이나 문제가 있으시면:

1. **이슈 트래커**: GitHub Issues에서 "realtime-saving" 라벨로 검색
2. **문서**: 이 README 파일과 코드 주석 참조
3. **테스트**: `test_realtime_saving.py` 스크립트로 기능 검증

---

**🎉 이제 학습 내용이 실시간으로 저장됩니다!** 

모든 기존 기능은 그대로 유지되며, 새로운 실시간 저장 시스템을 통해 더욱 안전하고 체계적인 학습이 가능합니다.
