# MediaPipe Pose landmark indices
# 11: left_shoulder, 12: right_shoulder
# 13: left_elbow,    14: right_elbow
# 15: left_wrist,    16: right_wrist
# 23: left_hip,      24: right_hip
# 25: left_knee,     26: right_knee
# 27: left_ankle,    28: right_ankle

EXERCISE_CONFIG = {
    "squat": {
        "display_name": "스쿼트",
        "keypoints": [23, 24, 25, 26, 27, 28],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 엉덩이~발목이 모두 보이도록 카메라를 조정해주세요",
    },
    "push_up": {
        "display_name": "푸시업",
        "keypoints": [11, 12, 13, 14, 15, 16, 23, 24],
        "min_visible": 6,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨~엉덩이가 모두 보이도록 카메라를 조정해주세요",
    },
    "bench_pressing": {
        "display_name": "벤치프레스",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면 또는 측면에서 상체 전체가 보이도록 카메라를 조정해주세요",
    },
    "pull_up": {
        "display_name": "풀업",
        "keypoints": [11, 12, 13, 14, 23, 24],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 어깨~엉덩이가 보이도록 카메라를 조정해주세요",
    },
    "front_raise": {
        "display_name": "프론트 레이즈",
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_visible": 5,
        "camera_hint": "front",
        "guide_message": "정면에서 양쪽 팔 전체가 보이도록 카메라를 조정해주세요",
    },
    "situp": {
        "display_name": "싯업",
        "keypoints": [11, 12, 23, 24, 25, 26],
        "min_visible": 5,
        "camera_hint": "side",
        "guide_message": "측면에서 어깨~무릎이 보이도록 카메라를 조정해주세요",
    },
    "jump_jack": {
        "display_name": "점핑잭",
        "keypoints": [11, 12, 23, 24, 25, 26, 27, 28],
        "min_visible": 7,
        "camera_hint": "front",
        "guide_message": "정면에서 전신이 모두 보이도록 카메라를 뒤로 이동해주세요",
    },
}

# 장비 → 가능한 운동 매핑
EQUIPMENT_EXERCISE_MAP = {
    "bench_press":  ["bench_pressing"],
    "pull_up_bar":  ["pull_up"],
    "dumbbell":     ["front_raise"],
    "squat_rack":   ["squat"],
    "floor":        ["push_up", "situp", "jump_jack", "squat"],
    "unknown":      ["push_up", "situp", "jump_jack", "squat", "front_raise", "pull_up", "bench_pressing"],
}

EQUIPMENT_DISPLAY_NAME = {
    "bench_press": "벤치프레스 랙",
    "pull_up_bar": "철봉",
    "dumbbell":    "덤벨 / 바벨",
    "squat_rack":  "스쿼트 랙",
    "floor":       "바닥 / 매트",
    "unknown":     "기구 미감지",
}

EQUIPMENT_EMOJI = {
    "bench_press": "🏋️",
    "pull_up_bar": "🔝",
    "dumbbell":    "💪",
    "squat_rack":  "🦵",
    "floor":       "🧘",
    "unknown":     "❓",
}
