# 실시간 운동 반복 횟수 카운팅 웹 애플리케이션

웹캠으로 운동 동작을 분석하여 반복 횟수를 자동으로 카운팅하는 실시간 웹 애플리케이션입니다.
YOLO-World 기반 장비 감지, OpenAI Vision API 기반 장비 분류, MediaPipe + Transformer 기반 포즈 분석을 결합하여 33가지 운동을 지원합니다.

---

## 주요 기능

- **자동 장비 감지**: YOLO-World 모델이 카메라에 비친 헬스 장비를 감지하고, GPT-4o-mini가 장비 종류를 분류합니다.
- **실시간 포즈 분석**: MediaPipe로 33개 신체 랜드마크를 추출하고, Transformer 기반 PoseRAC 모델로 동작을 분류합니다.
- **자동 반복 횟수 카운팅**: 이중 트리거 상태머신 + EMA 평활화를 통해 정확한 반복 횟수를 카운팅합니다.
- **카메라 가이드**: 필수 키포인트 가시성을 실시간으로 확인하고, 최적의 촬영 각도를 안내합니다.
- **33가지 운동 지원**: 15개 장비 카테고리에 걸쳐 맨몸 운동부터 머신 운동까지 지원합니다.

---

## 기술 스택

| 항목 | 기술 |
|------|------|
| 웹 프레임워크 | FastAPI + WebSocket (비동기) |
| 프론트엔드 | 단일 SPA (HTML/CSS/JS) + WebRTC |
| 포즈 감지 | MediaPipe Pose (33개 랜드마크) |
| 동작 분류 | Transformer 기반 PoseRAC (104차원 입력) |
| 장비 감지 | YOLO-World v2 |
| 장비 분류 | OpenAI GPT-4o-mini Vision API |
| 배포 | Docker + Nginx |

---

## 시스템 구조

```
웹 브라우저 (WebRTC 카메라)
        │ Base64 프레임 → WebSocket
        ↓
FastAPI 서버 (app.py)
  ├─ detecting  → YOLO-World: 바운딩 박스 감지
  ├─ capturing  → OpenAI Vision API: 장비 분류
  ├─ setup      → CameraGuide: 키포인트 가시성 확인
  └─ counting   → MediaPipe → PoseRAC → Action_trigger: 횟수 카운팅
        │ JSON 응답
        ↓
실시간 UI 업데이트
```

### 처리 흐름 (Phase)

```
detecting → confirming → selecting → setup → counting
```

1. **detecting**: 카메라에서 헬스 장비 바운딩 박스 감지
2. **confirming**: LLM이 장비 종류 분류 후 사용자 확인
3. **selecting**: 해당 장비에서 가능한 운동 목록 선택
4. **setup**: 카메라 각도 및 키포인트 가시성 안내
5. **counting**: 실시간 반복 횟수 카운팅

---

## 파일 구조

```
Real-time-pose-counting-web-application/
├── app.py                      # FastAPI 서버, WebSocket 핸들러
├── session.py                  # WebSocket 세션 상태 관리
├── model.py                    # PoseRAC 모델, Action_trigger 정의
├── pose_counter.py             # MediaPipe 포즈 감지 + 횟수 카운팅
├── exercise_config.py          # 운동 설정, 장비-운동 매핑
├── equipment_detector.py       # YOLO-World 기반 장비 감지
├── camera_guide.py             # 카메라 가이드 (키포인트 가시성)
├── llm_identifier.py           # OpenAI Vision API 장비 분류
├── utils.py                    # 유틸리티 함수
├── requirements_web.txt        # Python 의존성
├── RepCount_pose_config.yaml   # PoseRAC 모델 설정
├── best_weights.pth            # PoseRAC 학습된 가중치
├── yolov8s-worldv2.pt          # YOLO-World 모델 가중치
├── all_action.csv              # 운동 액션 라벨
├── static/
│   └── real_time.html          # 프론트엔드 SPA
├── Dockerfile
├── docker-compose.yml
└── nginx/                      # Nginx 설정 (배포용)
```

---

## 설치 및 실행

### 요구사항

- Python 3.8 이상 (3.9+ 권장)
- CUDA 지원 GPU (선택사항, CPU로도 동작)
- OpenAI API Key (장비 자동 분류 기능 사용 시)

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd Real-time-pose-counting-web-application

# 가상환경 생성 및 활성화
conda create -n re-rac python=3.9
conda activate re-rac

# 의존성 설치
pip install -r requirements_web.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API Key를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

> API Key가 없으면 장비 자동 분류 기능이 비활성화되고, 장비를 "unknown"으로 처리합니다.

### 3. 모델 가중치 확인

다음 파일이 프로젝트 루트에 있어야 합니다.

- `best_weights.pth` — PoseRAC 학습 가중치
- `yolov8s-worldv2.pt` — YOLO-World 모델 가중치

### 4. 실행

```bash
python app.py
```

브라우저에서 `http://localhost:8000` 접속

---

## Docker를 이용한 실행

```bash
# 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d
```

---

## 지원 운동 목록

### 모델 학습 운동 (7종)

| 운동명 | 설명 |
|--------|------|
| squat | 스쿼트 |
| push_up | 푸시업 |
| situp | 싯업 |
| jump_jack | 점프잭 |
| pull_up | 풀업 |
| front_raise | 프론트 레이즈 |
| bench_pressing | 벤치프레스 |

### 매핑 지원 운동 (26종)

학습된 7가지 운동과 가장 유사한 동작으로 자동 매핑되어 카운팅합니다.

| 분류 | 운동 목록 |
|------|---------|
| 맨몸 | lunge, jump_jack |
| 철봉/딥바 | leg_raises, dip |
| 바벨 | deadlift, barbell_row, ohp |
| 레그 머신 | leg_press, leg_curl, leg_extension |
| 케이블 머신 | cable_fly, cable_curl, cable_pushdown, cable_lateral_raise |
| 기타 머신 | lat_pull_down, seated_row, pec_deck_fly, machine_ohp |
| 자유 중량 | lateral_raises, kettlebell_swing, barbell_arm_curl |
| 유산소/기타 | rowing |

---

## 지원 장비 (15종)

| 장비 | 한글명 |
|------|--------|
| bench_press | 벤치프레스 |
| squat_rack | 스쿼트 랙 |
| smith_machine | 스미스 머신 |
| pull_up_bar | 철봉 |
| dip_bar | 딥 바 |
| lat_machine | 랫 풀다운 머신 |
| cable_machine | 케이블 머신 |
| leg_machine | 레그 머신 |
| pec_deck | 펙덱 머신 |
| shoulder_machine | 숄더 머신 |
| dumbbell | 덤벨 |
| kettlebell | 케틀벨 |
| cardio_machine | 유산소 기구 |
| abs_station | 복근 운동 기구 |
| floor | 맨바닥 (맨몸 운동) |

---

## 모델 상세

### PoseRAC (Transformer 기반 동작 분류)

- **입력**: 104차원 피처 벡터
  - 99차원: MediaPipe 33개 랜드마크 xyz 좌표 (정규화)
  - 5차원: 주요 관절 각도 (팔꿈치, 어깨, 엉덩이, 무릎, 발목)
- **구조**: TransformerEncoder (6층, 8헤드) + FC Layer
- **출력**: 0~1 확률값 (동작 수행 여부)

### Action_trigger (반복 횟수 카운팅)

```
pose_score > 0.717  →  동작 시작 감지
pose_score < 0.300  →  동작 완료 → 횟수 +1
```

EMA(지수이동평균, momentum=0.4)로 노이즈를 제거합니다.

---

## 주요 설정값 (RepCount_pose_config.yaml)

```yaml
PoseRAC:
  dim: 104              # 입력 피처 차원
  heads: 8              # Transformer 어텐션 헤드 수
  enc_layer: 6          # Transformer 인코더 레이어 수
  learning_rate: 0.001

Action_trigger:
  enter_threshold: 0.717    # 동작 시작 기준
  exit_threshold: 0.30      # 동작 완료 기준
  momentum: 0.4             # EMA 평활화 계수
```

---

## 문제 해결

### Python 버전 오류 (`TypeError: 'type' object is not subscriptable`)

Python 3.8 이하에서 발생하는 오류입니다. Python 3.9 이상으로 업그레이드하거나,
`exercise_config.py` 상단에 `from typing import Dict`를 추가하고 `dict[str, str]`을 `Dict[str, str]`로 변경하세요.

### OpenAI API Key 미설정

API Key 없이 실행하면 장비가 자동으로 "unknown"으로 분류되며, 모든 운동이 수동 선택 가능합니다.

### 카메라가 인식되지 않는 경우

브라우저에서 카메라 권한을 허용했는지 확인하세요. HTTPS 또는 localhost 환경에서만 WebRTC가 동작합니다.

---

## 라이선스

본 프로젝트는 연구 및 개인 사용 목적으로 제작되었습니다.
