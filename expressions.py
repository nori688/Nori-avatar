"""Библиотека пресетов выражений лица (Blendshapes) и математика их смешивания.

Гарантирует анатомически корректные позы и исключает перекосы лица.
Все параметры нормализованы в диапазоны модели:
- *_brow_y: -1.0 (вверх / удивление) ... +1.0 (вниз / хмурость)
- *_eye_x:  -1.0 (влево для зрителя) ... +1.0 (вправо для зрителя)
- *_eye_y:  -1.0 (вверх) ... +1.0 (вниз)
- *_eye_open: 0.0 (закрыт) ... 1.0 (распахнут)
- mouth_x:  -1.0 (влево) ... +1.0 (вправо)
- mouth_open: 0.0 (сомкнут) ... 1.0 (распахнут настежь)
- face_angle: -1.0 (наклон вправо) ... +1.0 (наклон влево)
"""

from typing import Dict, List, Tuple


NEUTRAL = {
    "left_brow_y": 0.0,
    "right_brow_y": 0.0,
    "left_eye_x": 0.0,
    "left_eye_y": 0.0,
    "right_eye_x": 0.0,
    "right_eye_y": 0.0,
    "left_eye_open": 1.0,
    "right_eye_open": 1.0,
    "mouth_x": 0.0,
    "mouth_open": 0.12,
    "face_angle": 0.0,
    "head_x": 0.0,
    "head_y": 0.0,
    "eye_wide": 0.0,
    "mouth_smile": 0.0,
    "mouth_wide": 0.0,
}

# Каталог эталонных пресетов
PRESETS: Dict[str, Dict[str, float]] = {
    "neutral": NEUTRAL.copy(),

    # Позитивные эмоции
    "smile": {
        "left_brow_y": -0.35,
        "right_brow_y": -0.35,
        "left_eye_open": 0.85,
        "right_eye_open": 0.85,
        "mouth_open": 0.50,
        "face_angle": 0.08,
        "mouth_smile": 0.55,
    },
    "joy": {
        "left_brow_y": -0.50,
        "right_brow_y": -0.50,
        "left_eye_open": 0.18,
        "right_eye_open": 0.18,
        "mouth_open": 0.85,
        "face_angle": 0.12,
        "mouth_smile": 0.8,
    },
    "laugh": {
        "left_brow_y": -0.60,
        "right_brow_y": -0.60,
        "left_eye_open": 0.10,
        "right_eye_open": 0.10,
        "mouth_open": 0.95,
        "face_angle": 0.18,
        "mouth_smile": 0.9,
    },

    # Негативные эмоции
    "sad": {
        "left_brow_y": 0.45,
        "right_brow_y": 0.45,
        "left_eye_y": 0.35,
        "right_eye_y": 0.35,
        "left_eye_open": 0.75,
        "right_eye_open": 0.75,
        "mouth_open": 0.08,
        "face_angle": -0.08,
        "mouth_smile": -0.55,
    },
    "angry": {
        "left_brow_y": 0.65,
        "right_brow_y": 0.65,
        "left_eye_y": 0.15,
        "right_eye_y": 0.15,
        "left_eye_open": 0.85,
        "right_eye_open": 0.85,
        "mouth_x": -0.15,
        "mouth_open": 0.22,
        "face_angle": -0.12,
        "mouth_smile": -0.35,
    },
    "disgust": {
        "left_brow_y": 0.40,
        "right_brow_y": 0.55,
        "left_eye_open": 0.60,
        "right_eye_open": 0.60,
        "mouth_x": 0.25,
        "mouth_open": 0.18,
        "face_angle": -0.15,
        "mouth_smile": -0.45,
    },

    # Удивление / Шок
    "surprise": {
        "left_brow_y": -0.80,
        "right_brow_y": -0.80,
        "left_eye_y": -0.15,
        "right_eye_y": -0.15,
        "left_eye_open": 1.0,
        "right_eye_open": 1.0,
        "mouth_open": 0.85,
        "face_angle": 0.04,
        "eye_wide": 1.0,
    },
    "shock": {
        "left_brow_y": -0.90,
        "right_brow_y": -0.90,
        "left_eye_open": 1.0,
        "right_eye_open": 1.0,
        "mouth_open": 1.0,
        "face_angle": 0.0,
        "eye_wide": 1.0,
    },

    # Игровые / Составные реакции
    "wink_left": {
        "left_eye_open": 0.0,
        "right_eye_open": 0.90,
        "left_brow_y": -0.30,
        "right_brow_y": 0.10,
        "mouth_x": 0.20,
        "mouth_open": 0.42,
        "face_angle": -0.08,
    },
    "wink_right": {
        "left_eye_open": 0.90,
        "right_eye_open": 0.0,
        "left_brow_y": 0.10,
        "right_brow_y": -0.30,
        "mouth_x": -0.20,
        "mouth_open": 0.42,
        "face_angle": 0.08,
    },
    "smug": {
        "left_brow_y": -0.35,
        "right_brow_y": 0.40,
        "left_eye_open": 0.75,
        "right_eye_open": 0.75,
        "mouth_x": 0.35,
        "mouth_open": 0.20,
        "face_angle": 0.15,
    },
    "skeptical": {
        "left_brow_y": -0.45,
        "right_brow_y": 0.45,
        "left_eye_open": 0.80,
        "right_eye_open": 0.80,
        "mouth_x": -0.25,
        "mouth_open": 0.15,
        "face_angle": -0.10,
    },
    "bored": {
        "left_brow_y": 0.15,
        "right_brow_y": 0.15,
        "left_eye_open": 0.40,
        "right_eye_open": 0.40,
        "left_eye_y": 0.25,
        "right_eye_y": 0.25,
        "mouth_open": 0.12,
        "face_angle": -0.10,
    },
    "sleepy": {
        "left_brow_y": 0.20,
        "right_brow_y": 0.20,
        "left_eye_open": 0.25,
        "right_eye_open": 0.25,
        "left_eye_y": 0.30,
        "right_eye_y": 0.30,
        "mouth_open": 0.10,
        "face_angle": -0.15,
    },

    # Направления взгляда и головы
    "look_left": {
        "left_eye_x": -0.85,
        "right_eye_x": -0.85,
        "face_angle": -0.25,
        "head_x": -0.35,
    },
    "look_right": {
        "left_eye_x": 0.85,
        "right_eye_x": 0.85,
        "face_angle": 0.25,
        "head_x": 0.35,
    },
    "look_up": {
        "left_eye_y": -0.85,
        "right_eye_y": -0.85,
        "left_brow_y": -0.20,
        "right_brow_y": -0.20,
        "head_y": -0.30,
    },
    "look_down": {
        "left_eye_y": 0.85,
        "right_eye_y": 0.85,
        "left_brow_y": 0.15,
        "right_brow_y": 0.15,
        "head_y": 0.45,
    },

    # Моргание / Глаза
    "blink": {
        "left_eye_open": 0.0,
        "right_eye_open": 0.0,
    },

    # Разговор
    "talk": {
        "mouth_open": 0.65,
        "mouth_x": 0.05,
    },
}

ZERO_TO_ONE_PARAMS = {"left_eye_open", "right_eye_open", "mouth_open", "eye_wide"}


def clamp(val: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, val))


def blend_presets(active_presets, base_name: str = "neutral") -> Dict[str, float]:
    """Смешивает пресеты и прямые значения параметров в одну позу.

    Элемент ``("smile", 0.8)`` — пресет с весом.
    Элемент ``("head_y", 0.75)`` — прямое значение параметра; используется
    в шагах таймлайна, когда нужно точное положение, а не эмоция.
    """
    result = dict(PRESETS.get(base_name, NEUTRAL))
    overrides = []

    for name, value in active_presets:
        if name in PRESETS:
            weight = clamp(float(value), 0.0, 1.5)
            preset = PRESETS[name]
            for param, target_val in preset.items():
                if param not in NEUTRAL:
                    continue
                result[param] = result.get(param, NEUTRAL[param]) + (
                    target_val - NEUTRAL[param]
                ) * weight
        elif name in NEUTRAL:
            overrides.append((name, float(value)))

    for param, value in overrides:
        result[param] = value

    for param in result:
        if param in ZERO_TO_ONE_PARAMS:
            result[param] = clamp(result[param], 0.0, 1.0)
        else:
            result[param] = clamp(result[param], -1.0, 1.0)

    return result
