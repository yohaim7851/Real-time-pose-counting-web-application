from typing import Dict

# MediaPipe Pose landmark indices
# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow,    14: right_elbow
# 15: left_wrist,    16: right_wrist
# 23: left_hip,      24: right_hip
# 25: left_knee,     26: right_knee
# 27: left_ankle,    28: right_ankle

EXERCISE_CONFIG = {
    # ── 맨몸 운동 ──────────────────────────────────────────────────────────
    "squat": {
        "display_name": "스쿼트",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이부터 발목이 모두 보이도록 카메라를 조정해주세요",
    },
    "push_up": {
        "display_name": "푸시업",
        "keypoints": [11, 12, 13, 14, 15, 16, 23, 24],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 엉덩이가 모두 보이도록 카메라를 조정해주세요",
    },
    "situp": {
        "display_name": "싯업",
        "keypoints": [11, 12, 23, 24, 25, 26],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 무릎이 보이도록 카메라를 조정해주세요",
    },
    "lunge": {
        "display_name": "런지",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이부터 발목이 모두 보이도록 카메라를 조정해주세요",
    },
    # ── 철봉 / 딥바 운동 ────────────────────────────────────────────────────
    "pull_up": {
        "display_name": "풀업",
        "keypoints": [11, 12, 13, 14, 23, 24],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 엉덩이가 보이도록 카메라를 조정해주세요",
    },
    "leg_raises": {
        "display_name": "레그 레이즈",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 엉덩이부터 발목이 보이도록 카메라를 조정해주세요",
    },
    "dip": {
        "display_name": "딥",
        "keypoints": [11, 12, 13, 14, 15, 16, 23, 24],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 팔꿈치가 모두 보이도록 카메라를 조정해주세요",
    },
    # ── 벤치프레스 ─────────────────────────────────────────────────────────
    "bench_pressing": {
        "display_name": "벤치프레스",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면 또는 측면에서 상체 전체가 보이도록 카메라를 조정해주세요",
    },
    # ── 스쿼트랙 / 스미스머신 운동 ────────────────────────────────────────
    "deadlift": {
        "display_name": "데드리프트",
        "keypoints": [11, 12, 23, 24, 25, 26, 27, 28],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 발목이 모두 보이도록 카메라를 조정해주세요",
    },
    "barbell_row": {
        "display_name": "바벨 로우",
        "keypoints": [11, 12, 13, 14, 23, 24],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 엉덩이가 보이도록 카메라를 조정해주세요",
    },
    "ohp": {
        "display_name": "오버헤드 프레스",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 팔 전체가 보이도록 카메라를 조정해주세요",
    },
    # ── 랫 머신 / 케이블 머신 운동 ────────────────────────────────────────
    "lat_pull_down": {
        "display_name": "랫 풀다운",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 손목이 모두 보이도록 카메라를 조정해주세요",
    },
    "seated_row": {
        "display_name": "시티드 로우",
        "keypoints": [11, 12, 13, 14, 15, 16, 23, 24],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 엉덩이가 보이도록 카메라를 조정해주세요",
    },
    "cable_fly": {
        "display_name": "케이블 플라이",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양팔이 모두 보이도록 카메라를 조정해주세요",
    },
    "cable_curl": {
        "display_name": "케이블 컬",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 손목이 보이도록 카메라를 조정해주세요",
    },
    "cable_pushdown": {
        "display_name": "케이블 푸시다운",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 손목이 보이도록 카메라를 조정해주세요",
    },
    "cable_lateral_raise": {
        "display_name": "케이블 레터럴 레이즈",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양팔이 모두 보이도록 카메라를 조정해주세요",
    },
    # ── 레그 머신 운동 ────────────────────────────────────────────────────
    "leg_press": {
        "display_name": "레그 프레스",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이부터 발목이 보이도록 카메라를 조정해주세요",
    },
    "leg_curl": {
        "display_name": "레그 컬",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이부터 발목이 보이도록 카메라를 조정해주세요",
    },
    "leg_extension": {
        "display_name": "레그 익스텐션",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이부터 발목이 보이도록 카메라를 조정해주세요",
    },
    # ── 펙덱 / 체스트 머신 운동 ───────────────────────────────────────────
    "pec_deck_fly": {
        "display_name": "펙덱 플라이",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양팔이 모두 보이도록 카메라를 조정해주세요",
    },
    # ── 숄더 머신 운동 ─────────────────────────────────────────────────────
    "machine_ohp": {
        "display_name": "머신 숄더 프레스",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 손목이 보이도록 카메라를 조정해주세요",
    },
    # ── 덤벨 / 케틀벨 운동 ─────────────────────────────────────────────────
    "front_raise": {
        "display_name": "프론트 레이즈",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양쪽 팔 전체가 보이도록 카메라를 조정해주세요",
    },
    "lateral_raises": {
        "display_name": "레터럴 레이즈",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양팔이 모두 보이도록 카메라를 조정해주세요",
    },
    "barbell_arm_curl": {
        "display_name": "바벨 컬",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨부터 손목이 모두 보이도록 카메라를 조정해주세요",
    },
    "kettlebell_swing": {
        "display_name": "케틀벨 스윙",
        "keypoints": [11, 12, 23, 24, 25, 26, 27, 28],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 발목이 모두 보이도록 카메라를 조정해주세요",
    },
    # ── 로잉 머신 운동 ─────────────────────────────────────────────────────
    "rowing": {
        "display_name": "로잉",
        "keypoints": [11, 12, 13, 14, 15, 16, 23, 24],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨부터 엉덩이가 보이도록 카메라를 조정해주세요",
    },
}

# ── 사용자 선택 운동 → 모델 학습 운동 매핑 ─────────────────────────────────
# 모델이 학습한 운동 (15개):
#   barbellarmcurl(0), barbellrow(1), benchpress(2), deadlift(3),
#   lateralraises(4), legpress(5), legraise(6), letpulldown(7),
#   lunge(8), overheadpress(9), pull_up(10), push_up(11),
#   seatedrow(12), situp(13), squat(14)
EXERCISE_MODEL_MAP: Dict[str, str] = {
    # 직접 매핑
    "squat":               "squat",
    "push_up":             "push_up",
    "situp":               "situp",
    "lunge":               "lunge",
    "pull_up":             "pull_up",
    "lateral_raises":      "lateralraises",
    "barbell_arm_curl":    "barbellarmcurl",
    "bench_pressing":      "benchpress",
    "deadlift":            "deadlift",
    "barbell_row":         "barbellrow",
    "ohp":                 "overheadpress",
    "lat_pull_down":       "letpulldown",
    "seated_row":          "seatedrow",
    "leg_press":           "legpress",
    "leg_raises":          "legraise",
    # 유사 운동 매핑
    "dip":                 "push_up",      # 밀기 계열
    "pec_deck_fly":        "lateralraises",# 팔 벌리기 계열
    "front_raise":         "lateralraises",# 어깨 올리기 계열
    "machine_ohp":         "overheadpress",
    "cable_fly":           "lateralraises",
    "cable_curl":          "barbellarmcurl",
    "cable_pushdown":      "push_up",
    "cable_lateral_raise": "lateralraises",
    "leg_curl":            "legpress",     # 레그 머신 계열
    "leg_extension":       "legpress",     # 레그 머신 계열
    "kettlebell_swing":    "deadlift",     # 힙힌지 계열
    "rowing":              "seatedrow",    # 로잉 계열
}

# ── 장비 → 가능한 운동 매핑 ────────────────────────────────────────────────
EQUIPMENT_EXERCISE_MAP: dict = {
    "bench_press":      ["bench_pressing", "ohp"],
    "squat_rack":       ["squat", "deadlift", "barbell_row", "ohp"],
    "smith_machine":    ["squat", "bench_pressing", "ohp", "deadlift", "lunge"],
    "pull_up_bar":      ["pull_up", "leg_raises"],
    "dip_bar":          ["dip", "leg_raises", "pull_up"],
    "lat_machine":      ["lat_pull_down", "seated_row"],
    "cable_machine":    ["cable_fly", "cable_curl", "cable_pushdown", "cable_lateral_raise", "lat_pull_down", "seated_row"],
    "leg_machine":      ["leg_press", "leg_curl", "leg_extension"],
    "pec_deck":         ["pec_deck_fly", "bench_pressing"],
    "shoulder_machine": ["machine_ohp", "lateral_raises"],
    "dumbbell":         ["lateral_raises", "front_raise", "barbell_arm_curl", "lunge", "ohp"],
    "kettlebell":       ["kettlebell_swing", "barbell_arm_curl", "lateral_raises", "lunge"],
    "cardio_machine":   ["squat", "lunge", "rowing"],
    "abs_station":      ["leg_raises", "situp"],
    "floor":            ["push_up", "situp", "lunge", "squat"],
    "unknown":          [
        "push_up", "situp", "squat", "pull_up", "bench_pressing",
        "deadlift", "lat_pull_down", "ohp", "lunge", "lateral_raises",
        "barbell_arm_curl", "seated_row", "leg_press", "barbell_row",
        "leg_raises", "cable_fly", "kettlebell_swing", "dip",
    ],
}

EQUIPMENT_DISPLAY_NAME: dict = {
    "bench_press":      "벤치프레스 랙",
    "squat_rack":       "스쿼트 랙",
    "smith_machine":    "스미스 머신",
    "pull_up_bar":      "철봉",
    "dip_bar":          "딥바 / 평행봉",
    "lat_machine":      "랫 머신",
    "cable_machine":    "케이블 머신",
    "leg_machine":      "레그 머신",
    "pec_deck":         "펙덱 / 체스트 플라이 머신",
    "shoulder_machine": "숄더 프레스 머신",
    "dumbbell":         "덤벨 / 바벨",
    "kettlebell":       "케틀벨",
    "cardio_machine":   "유산소 기구",
    "abs_station":      "복근 기구",
    "floor":            "바닥 / 매트",
    "unknown":          "기구 미감지",
}

EQUIPMENT_EMOJI: dict = {
    "bench_press":      "🏋️",
    "squat_rack":       "🦵",
    "smith_machine":    "🔩",
    "pull_up_bar":      "🔝",
    "dip_bar":          "📊",
    "lat_machine":      "🔄",
    "cable_machine":    "🔗",
    "leg_machine":      "🦿",
    "pec_deck":         "🫁",
    "shoulder_machine": "🙆",
    "dumbbell":         "💪",
    "kettlebell":       "🫙",
    "cardio_machine":   "🏃",
    "abs_station":      "🧘",
    "floor":            "🟩",
    "unknown":          "❓",
}
