"""Система таймлайнов / последовательностей анимации (Keyframe Timeline System).

Позволяет создавать органичные многошаговые анимации (последовательности действий)
с ключевыми кадрами, плавными кривыми интерполяции (easing) и автоматическим возвратом.

Примеры использования:
    - «закати глаза»: глаза делают круговое движение (вверх -> вправо -> вниз -> влево -> центр)
    - «кивни медленно»: голова плавно опускается вниз с задержкой и поднимается
    - «закати глаза и кивни медленно»: составная цепочка из нескольких действий
"""

import math
from typing import Dict, List, Optional, Tuple

import expressions


# ============================================================================
# КРИВЫЕ ИНТЕРПОЛЯЦИИ (Easing Functions)
# ============================================================================

def ease_linear(t: float) -> float:
    return t

def ease_in_out_cubic(t: float) -> float:
    """Плавный разгон и плавное торможение (гладкие переходы)."""
    if t < 0.5:
        return 4.0 * t * t * t
    p = 2.0 * t - 2.0
    return 0.5 * p * p * p + 1.0

def ease_in_out_sine(t: float) -> float:
    """Мягкая синусоида — идеальна для круговых движений глаз и дыхания."""
    return 0.5 * (1.0 - math.cos(math.pi * t))

def ease_out_quad(t: float) -> float:
    """Быстрый старт с мягким торможением — для резких реакций (моргание, вздрагивание)."""
    return t * (2.0 - t)

def ease_in_quad(t: float) -> float:
    """Плавный старт с ускорением."""
    return t * t

EASING_FUNCTIONS = {
    "linear": ease_linear,
    "cubic": ease_in_out_cubic,
    "sine": ease_in_out_sine,
    "out_quad": ease_out_quad,
    "in_quad": ease_in_quad,
}


# ============================================================================
# ШАГИ И ПОСЛЕДОВАТЕЛЬНОСТИ
# ============================================================================

class KeyframeStep:
    """Один шаг в анимационной цепочке.

    Атрибуты:
        presets: список пресетов (имя, вес) целевого состояния шага
        duration: длительность перехода к этому состоянию (в секундах)
        easing: имя функции сглаживания ('cubic', 'sine', 'out_quad' и др.)
    """
    __slots__ = ("presets", "duration", "easing")

    def __init__(
        self,
        presets: List[Tuple[str, float]],
        duration: float = 0.3,
        easing: str = "cubic",
    ):
        self.presets = presets
        self.duration = max(0.01, duration)
        self.easing = easing


class Sequence:
    """Именованная последовательность анимационных шагов."""
    __slots__ = ("name", "steps", "total_duration", "return_to_neutral", "return_duration")

    def __init__(
        self,
        name: str,
        steps: List[KeyframeStep],
        return_to_neutral: bool = True,
        return_duration: float = 0.35,
    ):
        self.name = name
        self.steps = steps
        self.total_duration = sum(s.duration for s in steps)
        self.return_to_neutral = return_to_neutral
        self.return_duration = return_duration if return_to_neutral else 0.0


# ============================================================================
# БИБЛИОТЕКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# ============================================================================

SEQUENCES: Dict[str, Sequence] = {}


def _reg(name: str, steps: List[KeyframeStep], return_to_neutral: bool = True, return_duration: float = 0.35):
    SEQUENCES[name] = Sequence(name, steps, return_to_neutral, return_duration)


# 1. Закатывание глаз (Eye Roll) — непрерывное круговое движение глаз
_reg("eye_roll", [
    KeyframeStep([("look_up", 0.95)], 0.22, "sine"),
    KeyframeStep([("look_right", 0.85), ("look_up", 0.35)], 0.22, "sine"),
    KeyframeStep([("look_right", 0.60), ("look_down", 0.65)], 0.22, "sine"),
    KeyframeStep([("look_left", 0.85), ("look_down", 0.40)], 0.22, "sine"),
    KeyframeStep([("look_left", 0.70), ("look_up", 0.50)], 0.20, "sine"),
    KeyframeStep([("neutral", 1.0)], 0.25, "cubic"),
], return_to_neutral=True, return_duration=0.2)

# 2. Быстрый кивок (Nod)
_reg("nod", [
    KeyframeStep([("look_down", 0.50), ("head_y", 0.75)], 0.25, "cubic"),
    KeyframeStep([("look_down", 0.50), ("head_y", 0.75)], 0.10, "linear"),
    KeyframeStep([("neutral", 1.0), ("head_y", 0.0)], 0.30, "cubic"),
])

# 3. Медленный кивок (Slow Nod) — глубокий выразительный наклон головы и взгляда
_reg("slow_nod", [
    KeyframeStep([("look_down", 0.70), ("head_y", 0.95), ("left_eye_open", 0.85), ("right_eye_open", 0.85)], 0.55, "sine"),
    KeyframeStep([("look_down", 0.70), ("head_y", 0.95)], 0.20, "linear"),
    KeyframeStep([("neutral", 1.0), ("head_y", 0.0)], 0.55, "cubic"),
])

# 4. Покачивание головой «нет» (Head Shake)
_reg("head_shake", [
    KeyframeStep([("face_angle", -0.65), ("head_x", -0.45), ("look_left", 0.55)], 0.20, "sine"),
    KeyframeStep([("face_angle", 0.65), ("head_x", 0.45), ("look_right", 0.55)], 0.24, "sine"),
    KeyframeStep([("face_angle", -0.40), ("head_x", -0.25), ("look_left", 0.35)], 0.20, "sine"),
    KeyframeStep([("face_angle", 0.0), ("head_x", 0.0), ("neutral", 1.0)], 0.24, "cubic"),
])

# 5. Двойное моргание (Double Blink)
_reg("double_blink", [
    KeyframeStep([("blink", 1.0)], 0.09, "out_quad"),
    KeyframeStep([("neutral", 1.0)], 0.09, "out_quad"),
    KeyframeStep([("blink", 1.0)], 0.09, "out_quad"),
    KeyframeStep([("neutral", 1.0)], 0.12, "cubic"),
])

# 6. Медленное подмигивание с ухмылкой (Slow Wink)
_reg("slow_wink", [
    KeyframeStep([("wink_left", 0.95), ("smile", 0.40)], 0.35, "cubic"),
    KeyframeStep([("wink_left", 0.95), ("smile", 0.55)], 0.45, "sine"),
    KeyframeStep([("neutral", 1.0)], 0.40, "cubic"),
])

# 7. Тяжелый вздох (Sigh)
_reg("sigh", [
    KeyframeStep([("look_down", 0.50), ("sad", 0.35), ("head_y", 0.45)], 0.45, "sine"),
    KeyframeStep([("look_down", 0.50), ("talk", 0.50), ("sad", 0.30), ("head_y", 0.45)], 0.40, "sine"),
    KeyframeStep([("neutral", 1.0), ("head_y", 0.0)], 0.50, "cubic"),
])

# 8. Оглядывание по сторонам (Look Around)
_reg("look_around", [
    KeyframeStep([("look_left", 0.85), ("head_x", -0.50), ("face_angle", -0.30)], 0.35, "sine"),
    KeyframeStep([("look_left", 0.85), ("head_x", -0.50), ("face_angle", -0.30)], 0.18, "linear"),
    KeyframeStep([("look_right", 0.85), ("head_x", 0.50), ("face_angle", 0.30)], 0.45, "sine"),
    KeyframeStep([("look_right", 0.85), ("head_x", 0.50), ("face_angle", 0.30)], 0.18, "linear"),
    KeyframeStep([("neutral", 1.0), ("head_x", 0.0), ("face_angle", 0.0)], 0.40, "cubic"),
])

# 9. Задумчивость (Thinking)
_reg("thinking", [
    KeyframeStep([("look_up", 0.70), ("look_right", 0.50), ("face_angle", 0.45), ("head_y", -0.55), ("skeptical", 0.35)], 0.45, "sine"),
    KeyframeStep([("look_up", 0.60), ("look_right", 0.60), ("face_angle", 0.35), ("head_y", -0.40), ("skeptical", 0.40)], 0.65, "sine"),
    KeyframeStep([("neutral", 1.0), ("head_y", 0.0), ("face_angle", 0.0)], 0.45, "cubic"),
])

# 10. Радостный кивок / Приветствие (Happy Nod)
_reg("happy_nod", [
    KeyframeStep([("smile", 0.85), ("head_y", 0.65), ("look_down", 0.30)], 0.30, "cubic"),
    KeyframeStep([("smile", 0.90), ("head_y", -0.15)], 0.28, "cubic"),
    KeyframeStep([("smile", 0.85), ("head_y", 0.35)], 0.24, "cubic"),
    KeyframeStep([("smile", 0.75), ("head_y", 0.0)], 0.35, "cubic"),
])

# 11. Смущение (Embarrassed)
_reg("embarrassed", [
    KeyframeStep([("look_down", 0.55), ("smile", 0.35), ("head_y", 0.35), ("face_angle", -0.15)], 0.35, "sine"),
    KeyframeStep([("look_left", 0.45), ("smile", 0.50), ("head_y", 0.40), ("face_angle", -0.20)], 0.40, "sine"),
    KeyframeStep([("look_down", 0.60), ("smile", 0.45), ("head_y", 0.30)], 0.40, "cubic"),
])

# 12. Злобный прищур (Angry Squint)
_reg("angry_squint", [
    KeyframeStep([("angry", 0.70), ("head_y", 0.20)], 0.30, "in_quad"),
    KeyframeStep([("angry", 1.00), ("blink", 0.45), ("head_y", 0.25)], 0.30, "cubic"),
    KeyframeStep([("angry", 0.85), ("head_y", 0.15)], 0.35, "sine"),
])


# ============================================================================
# ПРОИГРЫВАТЕЛЬ ТАЙМЛАЙНА
# ============================================================================

class Timeline:
    """Точный покадровый проигрыватель анимационных цепочек."""

    def __init__(
        self,
        sequence: Sequence,
        start_raw: Optional[Dict[str, float]] = None,
        on_finish_raw: Optional[Dict[str, float]] = None,
    ):
        self.sequence = sequence
        self.elapsed = 0.0
        self.finished = False

        # Нормализованные позы
        self._start_raw = start_raw.copy() if start_raw else expressions.NEUTRAL.copy()
        self._finish_raw = on_finish_raw.copy() if on_finish_raw else expressions.NEUTRAL.copy()

        # Предвычисляем целевые позы для каждого шага
        self._step_targets: List[Dict[str, float]] = []
        for step in sequence.steps:
            self._step_targets.append(expressions.blend_presets(step.presets))

        # Общее время всей анимации (шаги + возврат)
        self._total_time = sequence.total_duration + sequence.return_duration

    @property
    def finish_raw(self) -> Dict[str, float]:
        """Поза, к которой нужно вернуться после анимации."""
        return self._finish_raw

    def advance(self, delta_time: float) -> Optional[Dict[str, float]]:
        """Продвигает анимацию вперед и возвращает нормализованные параметры кадра."""
        if self.finished:
            return None

        self.elapsed += delta_time

        # Если анимация подошла к концу
        if self.elapsed >= self._total_time:
            self.finished = True
            if self.sequence.return_to_neutral:
                return self._finish_raw.copy()
            else:
                return self._step_targets[-1].copy() if self._step_targets else self._finish_raw.copy()

        # Фаза основных шагов
        if self.elapsed < self.sequence.total_duration:
            accum_time = 0.0
            prev_pose = self._start_raw

            for i, step in enumerate(self.sequence.steps):
                step_end = accum_time + step.duration
                if self.elapsed < step_end or i == len(self.sequence.steps) - 1:
                    # Мы внутри этого шага
                    t_in_step = self.elapsed - accum_time
                    progress = max(0.0, min(1.0, t_in_step / step.duration))
                    easing_fn = EASING_FUNCTIONS.get(step.easing, ease_in_out_cubic)
                    eased = easing_fn(progress)
                    target_pose = self._step_targets[i]
                    return self._lerp(prev_pose, target_pose, eased)

                prev_pose = self._step_targets[i]
                accum_time = step_end

        # Фаза плавного возврата в финишную позу
        if self.sequence.return_to_neutral and self.sequence.return_duration > 0.0:
            return_elapsed = self.elapsed - self.sequence.total_duration
            progress = max(0.0, min(1.0, return_elapsed / self.sequence.return_duration))
            eased = ease_in_out_cubic(progress)
            last_pose = self._step_targets[-1] if self._step_targets else self._start_raw
            return self._lerp(last_pose, self._finish_raw, eased)

        return self._finish_raw.copy()

    @staticmethod
    def _lerp(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
        """Плавная линейная интерполяция между двумя словарями нормализованных параметров."""
        res = {}
        for key in expressions.NEUTRAL:
            va = a.get(key, expressions.NEUTRAL[key])
            vb = b.get(key, expressions.NEUTRAL[key])
            res[key] = va + (vb - va) * t
        return res


# ============================================================================
# СКЛЕЙКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# ============================================================================

def chain_sequences(*names: str) -> Optional[Sequence]:
    """Объединяет несколько именованных последовательностей в единую цепочку."""
    valid = [name for name in names if name in SEQUENCES]
    combined_steps: List[KeyframeStep] = []

    for index, name in enumerate(valid):
        seq = SEQUENCES[name]
        for step in seq.steps:
            combined_steps.append(KeyframeStep(
                presets=list(step.presets),
                duration=step.duration,
                easing=step.easing,
            ))
        # Короткая пауза между действиями, но не после последнего
        if index < len(valid) - 1 and combined_steps:
            combined_steps.append(KeyframeStep(
                presets=list(combined_steps[-1].presets),
                duration=0.08,
                easing="linear",
            ))

    if not combined_steps:
        return None

    return Sequence(
        "chain_" + "_".join(valid), combined_steps,
        return_to_neutral=True, return_duration=0.35,
    )


def get_sequence(name: str) -> Optional[Sequence]:
    return SEQUENCES.get(name)


def list_sequences() -> List[str]:
    return list(SEQUENCES.keys())
