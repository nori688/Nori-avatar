"""Главный модуль управления 2D-аватаром Nori.

Решение о мимике принимает локальная Qwen (Ollama). Запрос уходит в фоновый
поток, поэтому окно и анимация не замирают даже на медленной машине.
Если модель недоступна или не успела ответить — используется быстрый
локальный семантический разбор (semantic_router).

Аватары:
1. NoriAvatar — слои из py.psd, scale 0.35 (по умолчанию)
2. ClassicAvatar — пиксельный аниме-арт
3. PixelSmiley — процедурный смайлик 96x96

Клавиши: Tab / 1 2 3 — аватар, F2 — нейтраль, H — чёлка, Esc — выход.
"""

import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pygame

import expressions
import fallbacks
from qwen_client import create_client
from semantic_router import SemanticRouter
from timeline import Timeline, SEQUENCES, chain_sequences

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
BACKGROUND_COLOR = (40, 48, 64)

PROJECT_DIR = Path(__file__).resolve().parent
NORI_MANIFEST_PATH = PROJECT_DIR / "assets" / "nori" / "manifest.json"
QWEN_CACHE_PATH = PROJECT_DIR / "models" / "qwen_cache.json"

# Сколько ждём Qwen, прежде чем применить локальный разбор
QWEN_TIMEOUT = 25.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def load_layer(filename, fallback_size, fallback_draw):
    """Ищет PNG в корне проекта и в папке assets, иначе рисует заглушку."""
    candidate_paths = [
        PROJECT_DIR / filename,
        PROJECT_DIR / "assets" / filename,
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                return pygame.image.load(str(path)).convert_alpha()
            except pygame.error:
                continue

    surface = pygame.Surface(fallback_size, pygame.SRCALPHA)
    fallback_draw(surface)
    return surface


# ==============================================================================
# МОЗГ: QWEN + ЛОКАЛЬНЫЙ РЕЗЕРВ
# ==============================================================================

class AvatarBrain:
    """Асинхронная связка Qwen (основная) и локального роутера (резерв)."""

    def __init__(self, qwen_cache_path):
        self.local = SemanticRouter()
        self.qwen = create_client(qwen_cache_path)
        self.pending_text = None
        self.pending_since = 0.0

    @property
    def backend_label(self):
        if self.qwen.model:
            return f"{self.qwen.label} · {self.qwen.model}"
        if self.qwen.checked:
            return f"локальный разбор · {self.qwen.label} недоступен"
        return f"{self.qwen.label} · подключение…"

    @property
    def busy(self):
        return self.pending_text is not None

    def ask(self, text):
        """Возвращает готовый результат или None, если Qwen ещё думает."""
        hit = self.qwen.cached(text)
        if hit is not None:
            self.pending_text = None
            return self._pack(text, hit, f"{self.qwen.label} · кэш")

        # Пока Qwen подключается, запрос уже можно ставить в очередь
        if self.qwen.enabled and (not self.qwen.checked or self.qwen.is_available):
            self.qwen.submit(text)
            self.pending_text = text
            self.pending_since = time.time()
            return None

        # Мгновенный локальный ответ отменяет ожидающий запрос Qwen,
        # иначе его поздний ответ перезапишет новую команду.
        self.pending_text = None
        return self._local(text)

    def poll(self):
        item = self.qwen.poll()
        if item is not None:
            text, result = item
            if self.pending_text is None or text != self.pending_text:
                return None  # устаревший ответ на отменённую команду
            self.pending_text = None
            if result is None:
                return self._local(text, note="Qwen не ответила")
            return self._pack(text, result, self.qwen.label)

        if self.pending_text and time.time() - self.pending_since > QWEN_TIMEOUT:
            text = self.pending_text
            self.pending_text = None
            return self._local(text, note="Qwen не ответила вовремя")

        return None

    @staticmethod
    def _pack(text, result, source):
        reactions = result.get("reactions") or []
        sequences = result.get("sequence") or []
        params = expressions.blend_presets(reactions) if reactions else None
        if reactions:
            desc = ", ".join(f"{name}:{weight:.2f}" for name, weight in reactions)
        else:
            desc = "чистая анимация"
        return {
            "text": text,
            "params": params,
            "seqs": sequences or None,
            "desc": desc,
            "source": source,
        }

    def _local(self, text, note=""):
        params, desc, _score, seqs = self.local.route(text)
        if note:
            desc = f"{desc} · {note}"
        return {
            "text": text,
            "params": params,
            "seqs": seqs,
            "desc": desc,
            "source": "локально",
        }


# ==============================================================================
# БАЗОВЫЙ КЛАСС АВАТАРА
# ==============================================================================

class BaseAvatar:
    """Базовый класс: сглаживание, таймлайны, естественное моргание."""

    NAME = "Base"

    DECAY_RATES = {
        "left_eye_x": 5.5,
        "left_eye_y": 5.5,
        "right_eye_x": 5.5,
        "right_eye_y": 5.5,
        "left_eye_open": 7.0,
        "right_eye_open": 7.0,
        "left_brow_y": 4.5,
        "right_brow_y": 4.5,
        "mouth_x": 4.0,
        "mouth_open": 4.5,
        "face_angle": 3.0,
        "head_x": 3.0,
        "head_y": 3.0,
    }
    DEFAULT_DECAY = 4.0

    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.time = 0.0

        self.blink_timer = random.uniform(3.5, 6.0)
        self.is_blinking = False
        self.blink_progress = 0.0
        self.blink_factor = 1.0

        neutral = dict(expressions.NEUTRAL)
        self.parameters = neutral.copy()
        self.target_parameters = neutral.copy()
        self.raw_target = neutral.copy()

        self._active_timeline = None

        # --- жизнь: затухание эмоции и микродвижения глаз ---
        self.hold_time = 4.0          # сколько держим эмоцию до затухания
        self._emotion_timer = 0.0
        self._decaying = False
        self._gaze = [0.0, 0.0]       # саккады, в нормализованных единицах
        self._gaze_timer = random.uniform(0.8, 2.2)

    # ------------------------------------------------------------------
    @property
    def is_animating(self):
        return self._active_timeline is not None

    def eye_open(self, side):
        """Открытость глаза с учётом естественного моргания."""
        value = self.parameters.get(side + "_eye_open", 1.0) * self.blink_factor
        return clamp(value, 0.0, 1.0)

    # ------------------------------------------------------------------
    def apply_model_output(self, output, seq_names=None):
        """Применяет позу и, если нужно, запускает последовательность."""
        data = dict(output or expressions.NEUTRAL)
        for key, value in expressions.NEUTRAL.items():
            data.setdefault(key, value)
        data = {k: float(v) for k, v in data.items()}

        self.raw_target = data
        self._emotion_timer = self.hold_time
        self._decaying = False
        self._map_with_gaze()

        sequence = None
        if seq_names:
            if len(seq_names) == 1:
                sequence = SEQUENCES.get(seq_names[0])
            if sequence is None:
                sequence = chain_sequences(*seq_names)

        if sequence is not None:
            self._active_timeline = Timeline(
                sequence, start_raw=self.raw_target, on_finish_raw=self.raw_target
            )
        else:
            self._active_timeline = None

    def _map_with_gaze(self):
        """Пересчитывает позу, добавляя микродвижение глаз (саккады)."""
        data = dict(self.raw_target)
        dx, dy = self._gaze
        if dx:
            data["left_eye_x"] = data.get("left_eye_x", 0.0) + dx
            data["right_eye_x"] = data.get("right_eye_x", 0.0) + dx
        if dy:
            data["left_eye_y"] = data.get("left_eye_y", 0.0) + dy
            data["right_eye_y"] = data.get("right_eye_y", 0.0) + dy
        self.map_model_parameters(data)

    def start_sequence(self, sequence):
        self._active_timeline = Timeline(
            sequence, start_raw=self.raw_target, on_finish_raw=self.raw_target
        )

    def map_model_parameters(self, output):
        self.target_parameters = output.copy()

    def reset(self):
        self._active_timeline = None
        self.apply_model_output(expressions.NEUTRAL.copy())
        self.blink_factor = 1.0

    # ------------------------------------------------------------------
    def update(self, delta_time):
        self.time += delta_time

        # --- Естественное моргание (применяется при отрисовке) ---
        self.blink_timer -= delta_time
        if not self.is_blinking and self.blink_timer <= 0.0:
            self.is_blinking = True
            self.blink_progress = 0.0
            self.blink_timer = random.uniform(3.0, 6.5)

        if self.is_blinking:
            self.blink_progress += delta_time / 0.16
            if self.blink_progress >= 1.0:
                self.is_blinking = False
                self.blink_factor = 1.0
            else:
                self.blink_factor = 4.0 * (self.blink_progress - 0.5) ** 2
        else:
            self.blink_factor = 1.0

        # --- Жизнь: саккады и затухание эмоции ---
        if self._active_timeline is None:
            self._gaze_timer -= delta_time
            commanded = abs(self.raw_target.get("left_eye_x", 0.0)) > 0.5
            if self._gaze_timer <= 0.0 and not commanded:
                self._gaze[0] = random.uniform(-0.10, 0.10)
                self._gaze[1] = random.uniform(-0.06, 0.06)
                self._gaze_timer = random.uniform(1.2, 3.6)
            elif commanded:
                self._gaze[0] *= 0.9
                self._gaze[1] *= 0.9

            if self._emotion_timer > 0.0:
                self._emotion_timer -= delta_time
            elif not self._decaying:
                self._decaying = True

            if self._decaying:
                # Плавно возвращаемся к нейтрали, чтобы лицо не «залипало»
                blend = 1.0 - math.exp(-1.6 * delta_time)
                raw = self.raw_target
                for key, neutral_value in expressions.NEUTRAL.items():
                    if key in raw:
                        raw[key] = raw[key] + (neutral_value - raw[key]) * blend
            self._map_with_gaze()

        # --- Таймлайн: кадры уже сглажены easing, применяем напрямую ---
        timeline = self._active_timeline
        if timeline is not None:
            frame = timeline.advance(delta_time)
            if frame is not None:
                self.map_model_parameters(frame)
                self.parameters = dict(self.target_parameters)
            if timeline.finished:
                self._active_timeline = None
                self.map_model_parameters(timeline.finish_raw)
                self.parameters = dict(self.target_parameters)
            return

        # --- Обычное экспоненциальное сглаживание ---
        for name in list(self.parameters):
            target = self.target_parameters.get(name, self.parameters[name])
            current = self.parameters[name]
            decay = self.DECAY_RATES.get(name, self.DEFAULT_DECAY)
            smoothing = 1.0 - math.exp(-decay * delta_time)
            self.parameters[name] = current + (target - current) * smoothing

    def draw(self, screen):
        raise NotImplementedError


# ==============================================================================
# 0. LIVE2D-АВАТАР (запечённые состояния)
# ==============================================================================

class Live2DAvatar(BaseAvatar):
    """Персонаж Live2D, отрисованный нашим движком по снятому ригу.

    Ни SDK, ни .moc3 в рантайме: есть JSON с вершинами/UV/индексами и
    сэмплами деформации, поза считается как «нейтраль + дельты».
    """

    NAME = "Mao (Live2D rig)"
    RIG = PROJECT_DIR / "assets" / "live2d" / "mao" / "rig.json.gz"
    TEXTURE = PROJECT_DIR / "assets" / "live2d" / "mao" / "texture_orig.png"
    WIDTH, HEIGHT = 520, 720

    # нормализованный параметр -> (id в модели, множитель)
    MAPPING = {
        "mouth_open": ("ParamMouthOpenY", 1.0),
        "mouth_smile": ("ParamMouthForm", 1.0),
        "brow_l": ("ParamBrowLY", 1.0),
        "brow_r": ("ParamBrowRY", 1.0),
        "eye_l": ("ParamEyeLOpen", 1.0),
        "eye_r": ("ParamEyeROpen", 1.0),
        "angle_x": ("ParamAngleX", 30.0),
        "angle_y": ("ParamAngleY", 30.0),
        "angle_z": ("ParamAngleZ", 30.0),
    }

    def __init__(self, screen_size):
        super().__init__(screen_size)
        # у этого аватара свои каналы, а не набор Нори
        neutral = {
            "mouth_open": 0.12, "mouth_smile": 0.0,
            "brow_l": 0.0, "brow_r": 0.0,
            "eye_l": 1.0, "eye_r": 1.0,
            "angle_x": 0.0, "angle_y": 0.0, "angle_z": 0.0,
        }
        self.parameters = neutral.copy()
        self.target_parameters = neutral.copy()

        self.rig = None
        try:
            from nori_engine.live2d import Live2DRig

            self.rig = Live2DRig(self.RIG, self.TEXTURE)
            print(f"[Live2D] риг загружен: {self.rig.vertex_count} вершин, "
                  f"{len(self.rig.params)} параметров")
        except Exception as error:
            print(f"[Live2D] не удалось загрузить риг: {error}")

    def map_model_parameters(self, output):
        self.target_parameters = {
            "mouth_open": clamp(output["mouth_open"], 0.0, 1.0),
            "mouth_smile": clamp(output.get("mouth_smile", 0.0), -1.0, 1.0),
            "brow_l": clamp(output["left_brow_y"], -1.0, 1.0),
            "brow_r": clamp(output["right_brow_y"], -1.0, 1.0),
            "eye_l": clamp(output["left_eye_open"], 0.0, 1.0),
            "eye_r": clamp(output["right_eye_open"], 0.0, 1.0),
            "angle_x": clamp(output.get("head_x", 0.0), -1.0, 1.0),
            "angle_y": clamp(-output.get("head_y", 0.0), -1.0, 1.0),
            "angle_z": clamp(-output.get("face_angle", 0.0), -1.0, 1.0),
        }

    def cycle_fringe(self):
        return "—"

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        if self.rig is None:
            return
        p = self.parameters
        values = {
            "ParamMouthOpenY": p["mouth_open"],
            "ParamMouthForm": p["mouth_smile"],
            "ParamBrowLY": p["brow_l"],
            "ParamBrowRY": p["brow_r"],
            "ParamEyeLOpen": p["eye_l"] * self.blink_factor,
            "ParamEyeROpen": p["eye_r"] * self.blink_factor,
            "ParamAngleX": p["angle_x"] * 30.0,
            "ParamAngleY": p["angle_y"] * 30.0,
            "ParamAngleZ": p["angle_z"] * 30.0,
        }
        self.rig.set_params(values)
        frame = self.rig.render(self.WIDTH, self.HEIGHT)
        screen.blit(frame, ((self.screen_size[0] - self.WIDTH) // 2,
                            self.screen_size[1] - self.HEIGHT))


# ==============================================================================
# 1. LIVE2D / PSD АВАТАР (NORI)
# ==============================================================================

class NoriAvatar(BaseAvatar):
    """Аватар на слоях py.psd: лицо гнётся мешем на GPU, тело — трансформации."""

    NAME = "Nori (Live2D PSD)"
    FRINGE_MODES = ("полная", "правая прядь", "без чёлки")

    # Геометрия века
    EYE_CLOSE_TOP = 20
    EYE_CLOSE_BOTTOM = 18
    EYELID_LIFT_MAX = 20

    # Геометрия рта
    MOUTH_CLOSED_RATIO = 0.18
    MOUTH_REST = 0.12
    MOUTH_OPEN_GAIN = 1.45
    MOUTH_CLOSED = 0.18
    MOUTH_OPEN_MESH = 0.95
    MOUTH_SMILE_GAIN = 0.10
    MOUTH_WIDE_GAIN = 0.30

    def __init__(self, screen_size):
        super().__init__(screen_size)
        self.screen_head = (screen_size[0] // 2, 235)
        self.scale = 0.35

        self.layers_info = {}
        self.surfaces = {}
        self._scale_cache = {}
        self.fringe_mode = 0

        if NORI_MANIFEST_PATH.exists():
            import json

            with open(NORI_MANIFEST_PATH, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.layers_info = manifest.get("layers", {})
            for name, info in self.layers_info.items():
                path = PROJECT_DIR / "assets" / "nori" / info["file"]
                if not path.exists():
                    continue
                raw = pygame.image.load(str(path)).convert_alpha()
                width = max(1, round(raw.get_width() * self.scale))
                height = max(1, round(raw.get_height() * self.scale))
                self.surfaces[name] = pygame.transform.smoothscale(raw, (width, height))

        self.head_canvas = pygame.Surface((640, 640), pygame.SRCALPHA)
        self.head_center = (320, 320)

        self.gpu = None
        self.gpu_error = ""
        self._head_bbox = self._compute_head_bbox()
        self._init_gpu()

    # ------------------------------------------------------------------
    def _init_gpu(self):
        if os.environ.get("NORI_SHARED_GPU") == "1":
            print("[Nori] GPU подключается к общему оконному OpenGL-контексту")
            return
        if os.environ.get("NORI_GPU") != "1":
            print("[Nori] оконный рендер на CPU; GPU-путь включается через NORI_GPU=1")
            return
        try:
            from nori_engine.gpu import GPUCanvas

            canvas = GPUCanvas((640, 640))
            for name, surface in self.surfaces.items():
                canvas.upload(name, surface)
            self.gpu = canvas
            print(f"[Nori] рендер на GPU: {canvas.renderer}")
        except Exception as error:
            self.gpu = None
            self.gpu_error = str(error)
            print(f"[Nori] GPU недоступен, рисую на CPU: {error}")

    def _compute_head_bbox(self):
        """Границы головы на холсте: по ним читаем кадр с GPU."""
        # Тело и шея живут на экране, а не на холсте головы
        skip = {"body", "back_hair", "neck"}
        xs, ys = [], []
        for name in self.surfaces:
            if name in skip:
                continue
            x0, y0, w, h = self._layer_rect(name, self.head_center)
            xs.extend([x0, x0 + w])
            ys.extend([y0, y0 + h])
        if not xs:
            return (0, 0, 640, 640)
        pad = 0.12 * max(max(xs) - min(xs), max(ys) - min(ys))
        x0 = int(max(0, min(xs) - pad))
        y0 = int(max(0, min(ys) - pad))
        x1 = int(min(640, max(xs) + pad))
        y1 = int(min(640, max(ys) + pad))
        return (x0, y0, x1 - x0, y1 - y0)

    # ------------------------------------------------------------------
    def map_model_parameters(self, output):
        self.target_parameters = {
            "left_brow_y": output["left_brow_y"] * 15.0,
            "right_brow_y": output["right_brow_y"] * 15.0,
            "left_eye_x": output["left_eye_x"] * 14.0,
            "left_eye_y": output["left_eye_y"] * 9.0,
            "right_eye_x": output["right_eye_x"] * 14.0,
            "right_eye_y": output["right_eye_y"] * 9.0,
            "left_eye_open": output["left_eye_open"],
            "right_eye_open": output["right_eye_open"],
            "mouth_x": output["mouth_x"] * 12.0,
            "mouth_open": output["mouth_open"],
            "mouth_smile": output.get("mouth_smile", 0.0),
            "mouth_wide": output.get("mouth_wide", 0.0),
            "face_angle": -output["face_angle"] * 22.0,
            "head_x": output.get("head_x", 0.0) * 36.0,
            "head_y": output.get("head_y", 0.0) * 36.0,
            "eye_wide": clamp(output.get("eye_wide", 0.0), 0.0, 1.0),
        }

    def cycle_fringe(self):
        self.fringe_mode = (self.fringe_mode + 1) % len(self.FRINGE_MODES)
        return self.FRINGE_MODES[self.fringe_mode]

    # ------------------------------------------------------------------
    # Геометрия
    # ------------------------------------------------------------------
    def _layer_rect(self, name, center, offset=(0, 0)):
        rel = self.layers_info[name]["rel_center"]
        width, height = self.surfaces[name].get_size()
        # Округляем позицию, иначе слой встаёт на полпикселя и края «мылятся»
        cx = round(center[0] + rel[0] * self.scale + offset[0])
        cy = round(center[1] + rel[1] * self.scale + offset[1])
        return cx - width // 2, cy - height // 2, width, height

    @staticmethod
    def _grid(rect, cols, rows):
        x0, y0, width, height = rect
        xs = np.linspace(x0, x0 + width, cols, dtype="f4")
        ys = np.linspace(y0, y0 + height, rows, dtype="f4")
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx, gy], axis=-1).astype("f4")

    def _rotate(self, points, angle):
        if abs(angle) < 0.01:
            return points
        radians = math.radians(angle)
        ca, sa = math.cos(radians), math.sin(radians)
        cx, cy = self.head_center
        x = points[..., 0] - cx
        y = points[..., 1] - cy
        return np.stack([x * ca - y * sa + cx, x * sa + y * ca + cy], axis=-1).astype("f4")

    # ------------------------------------------------------------------
    # GPU-путь
    # ------------------------------------------------------------------
    def _gpu_layer(self, name, center, offset=(0, 0), angle=0.0, uv=(0.0, 0.0, 1.0, 1.0)):
        if self.gpu is None or name not in self.surfaces:
            return
        rect = self._layer_rect(name, center, offset)
        points = self._rotate(self._grid(rect, 2, 2), angle)
        self.gpu.draw(name, points, 2, 2, uv)

    def _gpu_cropped(self, name, center, offset, top, bottom, angle):
        if self.gpu is None or name not in self.surfaces:
            return
        x0, y0, width, height = self._layer_rect(name, center, offset)
        top = max(0.0, min(height, top))
        bottom = max(0.0, min(height - top, bottom))
        visible = height - top - bottom
        if visible <= 0.5:
            return
        rect = (x0, y0 + top, width, visible)
        uv = (0.0, top / height, 1.0, (height - bottom) / height)
        points = self._rotate(self._grid(rect, 2, 2), angle)
        self.gpu.draw(name, points, 2, 2, uv)

    def _gpu_mouth(self, center, p, angle):
        if self.gpu is None or "mouth" not in self.surfaces:
            return
        x0, y0, width, height = self._layer_rect(
            "mouth", center, offset=(round(p["mouth_x"]), 0)
        )
        cols = rows = 5
        points = self._grid((x0, y0, width, height), cols, rows)
        cx, cy = x0 + width / 2.0, y0 + height / 2.0

        openness = clamp(p["mouth_open"], 0.0, 1.0)
        smile = clamp(p.get("mouth_smile", 0.0), -1.0, 1.0)
        wide = clamp(p.get("mouth_wide", 0.0), -1.0, 1.0)

        scale_y = self.MOUTH_CLOSED + self.MOUTH_OPEN_MESH * openness
        dx = points[..., 0] - cx
        dy = points[..., 1] - cy
        u = dx / max(1.0, width / 2.0)
        v = dy / max(1.0, height / 2.0)

        nx = cx + dx * (1.0 + self.MOUTH_WIDE_GAIN * wide)
        ny = cy + dy * scale_y
        ny -= smile * self.MOUTH_SMILE_GAIN * height * np.abs(u) ** 1.5 * (
            1.0 - 0.4 * np.maximum(v, 0.0)
        )
        points = np.stack([nx, ny], axis=-1).astype("f4")
        self.gpu.draw("mouth", self._rotate(points, angle), cols, rows)

    def _render_head_gpu(self, p, hair_sway):
        gpu = self.gpu
        gpu.clear((0, 0, 0, 0))
        hc = self.head_center
        angle = p["face_angle"]

        left_open = self.eye_open("left")
        right_open = self.eye_open("right")
        left_top = (1.0 - left_open) * self.EYE_CLOSE_TOP
        left_bottom = (1.0 - left_open) * self.EYE_CLOSE_BOTTOM
        right_top = (1.0 - right_open) * self.EYE_CLOSE_TOP
        right_bottom = (1.0 - right_open) * self.EYE_CLOSE_BOTTOM
        lift = p.get("eye_wide", 0.0) * self.EYELID_LIFT_MAX

        self._gpu_layer("face_base", hc, angle=angle)

        self._gpu_cropped("left_eye_white", hc, (0, 0), left_top, left_bottom, angle)
        self._gpu_cropped("right_eye_white", hc, (0, 0), right_top, right_bottom, angle)
        self._gpu_cropped(
            "left_eye_iris", hc,
            (round(p["left_eye_x"]), round(p["left_eye_y"])),
            left_top, left_bottom, angle,
        )
        self._gpu_cropped(
            "right_eye_iris", hc,
            (round(p["right_eye_x"]), round(p["right_eye_y"])),
            right_top, right_bottom, angle,
        )

        self._gpu_layer("left_eyelid_lower", hc, (0, -round(left_bottom)), angle)
        self._gpu_layer("right_eyelid_lower", hc, (0, -round(right_bottom)), angle)
        self._gpu_layer(
            "left_eyelid_upper", hc,
            (0, round(left_top - left_open * lift)), angle,
        )
        self._gpu_layer(
            "right_eyelid_upper", hc,
            (0, round(right_top - right_open * lift)), angle,
        )

        self._gpu_layer("nose", hc, angle=angle)
        self._gpu_mouth(hc, p, angle)
        self._gpu_layer("left_brow", hc, (0, round(p["left_brow_y"])), angle)
        self._gpu_layer("right_brow", hc, (0, round(p["right_brow_y"])), angle)

        if self.fringe_mode == 0:
            self._gpu_layer("front_hair_left", hc, (round(hair_sway * 1.15), 0), angle)
            self._gpu_layer("front_hair_mid", hc, (round(hair_sway * 0.45), 0), angle)
            self._gpu_layer("front_hair_right", hc, (round(hair_sway * 0.8), 0), angle)
        elif self.fringe_mode == 1:
            self._gpu_layer("front_hair_right", hc, (round(hair_sway), 0), angle)

        x, y, w, h = self._head_bbox
        frame = self.gpu.read(viewport=self._head_bbox)
        return frame, (x - self.head_center[0], y - self.head_center[1])

    # ------------------------------------------------------------------
    # CPU-путь (запасной)
    # ------------------------------------------------------------------
    def _scaled(self, name, scale_y=1.0, anchor="center"):
        base = self.surfaces[name]
        if abs(scale_y - 1.0) < 0.02:
            return base, 0
        key = (name, round(scale_y, 2), anchor)
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        base_h = base.get_height()
        new_h = max(2, int(base_h * scale_y))
        surface = pygame.transform.smoothscale(base, (base.get_width(), new_h))
        delta = new_h - base_h
        shift = delta // 2 if anchor == "top" else (-delta // 2 if anchor == "bottom" else 0)
        self._scale_cache[key] = (surface, shift)
        return surface, shift

    def _blit_layer(self, dest, name, center, offset=(0, 0), scale_y=1.0, anchor="center"):
        if name not in self.surfaces or name not in self.layers_info:
            return
        rel = self.layers_info[name]["rel_center"]
        x = center[0] + round(rel[0] * self.scale) + offset[0]
        y = center[1] + round(rel[1] * self.scale) + offset[1]
        surface, shift = self._scaled(name, scale_y, anchor)
        dest.blit(surface, surface.get_rect(center=(x, y + shift)))

    def _blit_eye_part(self, dest, name, center, offset, hidden_top=0.0, hidden_bottom=0.0):
        if name not in self.surfaces or name not in self.layers_info:
            return
        rel = self.layers_info[name]["rel_center"]
        x = center[0] + round(rel[0] * self.scale) + offset[0]
        y = center[1] + round(rel[1] * self.scale) + offset[1]
        base = self.surfaces[name]
        height = base.get_height()
        top_cut = max(0, min(height, int(round(hidden_top))))
        bottom_cut = max(0, min(height - top_cut, int(round(hidden_bottom))))
        visible = height - top_cut - bottom_cut
        if visible <= 1:
            return
        surface = base if (top_cut == 0 and bottom_cut == 0) else base.subsurface(
            pygame.Rect(0, top_cut, base.get_width(), visible)
        )
        dest.blit(surface, surface.get_rect(midbottom=(x, y + height // 2 - bottom_cut)))

    def _draw_mouth_cpu(self, canvas, center, params, depth_x=0):
        x_offset = round(params["mouth_x"]) + depth_x
        openness = clamp(params["mouth_open"], 0.0, 1.0)
        ratio = self.MOUTH_CLOSED_RATIO + max(0.0, openness - self.MOUTH_REST) * self.MOUTH_OPEN_GAIN
        smile = clamp(params.get("mouth_smile", 0.0), -1.0, 1.0)
        x0, y0, width, height = self._layer_rect("mouth", center, offset=(x_offset, 0))
        if abs(smile) < 0.01:
            self._blit_layer(canvas, "mouth", center, offset=(x_offset, 0), scale_y=ratio, anchor="center")
            return
        # На CPU сглаживаем только наклоном: полноценный меш есть на GPU
        tilted = pygame.transform.rotate(self.surfaces["mouth"], 0)
        self._blit_layer(canvas, "mouth", center, offset=(x_offset, 0), scale_y=ratio, anchor="center")

    def _render_head_cpu(self, p, hair_sway):
        self.head_canvas.fill((0, 0, 0, 0))
        hc = self.head_center
        depth = p["face_angle"] * 0.42 + p["head_x"] * 0.10
        base_x = round(depth * 0.16)
        eye_x = round(depth * 0.52)
        feature_x = round(depth * 0.78)
        hair_x = round(-depth * 0.62)

        left_open = self.eye_open("left")
        right_open = self.eye_open("right")
        left_top = (1.0 - left_open) * self.EYE_CLOSE_TOP
        left_bottom = (1.0 - left_open) * self.EYE_CLOSE_BOTTOM
        right_top = (1.0 - right_open) * self.EYE_CLOSE_TOP
        right_bottom = (1.0 - right_open) * self.EYE_CLOSE_BOTTOM
        lift = p.get("eye_wide", 0.0) * self.EYELID_LIFT_MAX

        self._blit_layer(self.head_canvas, "face_base", hc, offset=(base_x, 0))
        self._blit_eye_part(self.head_canvas, "left_eye_white", hc, (eye_x, 0), left_top, left_bottom)
        self._blit_eye_part(self.head_canvas, "right_eye_white", hc, (eye_x, 0), right_top, right_bottom)
        self._blit_eye_part(
            self.head_canvas, "left_eye_iris", hc,
            (eye_x + round(p["left_eye_x"]), round(p["left_eye_y"])), left_top, left_bottom,
        )
        self._blit_eye_part(
            self.head_canvas, "right_eye_iris", hc,
            (eye_x + round(p["right_eye_x"]), round(p["right_eye_y"])), right_top, right_bottom,
        )
        self._blit_layer(self.head_canvas, "left_eyelid_lower", hc, offset=(eye_x, -round(left_bottom)))
        self._blit_layer(self.head_canvas, "right_eyelid_lower", hc, offset=(eye_x, -round(right_bottom)))
        self._blit_layer(
            self.head_canvas, "left_eyelid_upper", hc,
            offset=(eye_x, round(left_top - left_open * lift)),
        )
        self._blit_layer(
            self.head_canvas, "right_eyelid_upper", hc,
            offset=(eye_x, round(right_top - right_open * lift)),
        )
        self._blit_layer(self.head_canvas, "nose", hc, offset=(feature_x, 0))
        self._draw_mouth_cpu(self.head_canvas, hc, p, feature_x)
        self._blit_layer(self.head_canvas, "left_brow", hc, offset=(eye_x, round(p["left_brow_y"])))
        self._blit_layer(self.head_canvas, "right_brow", hc, offset=(eye_x, round(p["right_brow_y"])))

        if self.fringe_mode == 0:
            self._blit_layer(self.head_canvas, "front_hair_left", hc, offset=(hair_x + round(hair_sway * 1.15), 0))
            self._blit_layer(self.head_canvas, "front_hair_mid", hc, offset=(hair_x + round(hair_sway * 0.45), 0))
            self._blit_layer(self.head_canvas, "front_hair_right", hc, offset=(hair_x + round(hair_sway * 0.8), 0))
        elif self.fringe_mode == 1:
            self._blit_layer(self.head_canvas, "front_hair_right", hc, offset=(hair_x + round(hair_sway), 0))

        return pygame.transform.rotate(self.head_canvas, p["face_angle"])

    # ------------------------------------------------------------------
    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        p = self.parameters

        breath_y = math.sin(self.time * 2.2) * 2.5
        hair_sway = math.cos(self.time * 1.8) * 1.5

        body_center = (self.screen_head[0], self.screen_head[1] + breath_y * 0.5)
        detailed_body = all(
            name in self.surfaces
            for name in (
                "torso", "left_arm_lower", "right_arm_lower",
                "left_arm_upper", "right_arm_upper", "left_leg", "right_leg",
            )
        )
        if detailed_body:
            self._blit_layer(screen, "back_hair", body_center, offset=(0, -2))
            self._blit_layer(screen, "left_arm_lower", body_center, offset=(0, round(breath_y)))
            self._blit_layer(screen, "right_arm_lower", body_center, offset=(0, round(breath_y)))
            self._blit_layer(screen, "torso", body_center, offset=(0, round(breath_y)))
            self._blit_layer(screen, "left_arm_upper", body_center, offset=(0, round(breath_y)))
            self._blit_layer(screen, "right_arm_upper", body_center, offset=(0, round(breath_y)))
        else:
            self._blit_layer(screen, "back_hair", body_center, offset=(0, -2))
            self._blit_layer(screen, "body", body_center, offset=(0, round(breath_y)))
        self._blit_layer(screen, "neck", body_center, offset=(0, round(breath_y * 0.7)))

        head_pos = (
            self.screen_head[0] + p["head_x"],
            self.screen_head[1] + breath_y + p["head_y"],
        )

        if self.gpu is not None:
            frame, offset = self._render_head_gpu(p, hair_sway)
            # Кадр приходит premultiplied — pygame умеет так блитить
            screen.blit(
                frame, (head_pos[0] + offset[0], head_pos[1] + offset[1]),
                special_flags=pygame.BLEND_PREMULTIPLIED,
            )
        else:
            frame = self._render_head_cpu(p, hair_sway)
            screen.blit(frame, frame.get_rect(center=head_pos))


# ==============================================================================
# 2. КЛАССИЧЕСКИЙ ПИКСЕЛЬ-АРТ АВАТАР
# ==============================================================================

class ClassicAvatar(BaseAvatar):
    """Классический пиксельный аватар на спрайтах или процедурных фоллбэках."""

    NAME = "Classic (Pixel Anime)"

    def __init__(self, screen_size):
        super().__init__(screen_size)
        self.screen_center = (screen_size[0] // 2, screen_size[1] // 2 - 55)

        self.body = load_layer(
            "body.png", (416, 288), fallbacks.draw_reference_body_fallback
        )
        self.face = load_layer(
            "face.png", (320, 400), fallbacks.draw_reference_face_fallback
        )
        self.hair_front = load_layer(
            "hair_front.png", (320, 400), fallbacks.draw_reference_hair_front_fallback
        )
        brow = load_layer("brow.png", (60, 12), fallbacks.draw_reference_brow_fallback)
        eye = load_layer("eye.png", (72, 40), fallbacks.draw_reference_eye_fallback)
        self.mouth = load_layer(
            "mouth.png", (56, 32), fallbacks.draw_reference_mouth_fallback
        )

        self.left_brow = brow
        self.right_brow = pygame.transform.flip(brow, True, False)
        self.left_eye = eye
        self.right_eye = pygame.transform.flip(eye, True, False)

        self.base_positions = {
            "body": (self.screen_center[0], self.screen_center[1] + 170),
            "face": self.screen_center,
            "left_eye": (self.screen_center[0] - 43, self.screen_center[1] + 16),
            "right_eye": (self.screen_center[0] + 43, self.screen_center[1] + 16),
            "mouth": (self.screen_center[0], self.screen_center[1] + 62),
            "left_brow": (self.screen_center[0] - 43, self.screen_center[1] - 18),
            "right_brow": (self.screen_center[0] + 43, self.screen_center[1] - 18),
        }

        self.head_surface = pygame.Surface((440, 440), pygame.SRCALPHA)
        self.local_center = (220, 220)

    def map_model_parameters(self, output):
        self.target_parameters = {
            "left_brow_y": output["left_brow_y"] * 28.0,
            "right_brow_y": output["right_brow_y"] * 28.0,
            "left_eye_x": output["left_eye_x"] * 24.0,
            "left_eye_y": output["left_eye_y"] * 16.0,
            "right_eye_x": output["right_eye_x"] * 24.0,
            "right_eye_y": output["right_eye_y"] * 16.0,
            "left_eye_open": output["left_eye_open"],
            "right_eye_open": output["right_eye_open"],
            "mouth_x": output["mouth_x"] * 32.0,
            "mouth_open": output["mouth_open"],
            "face_angle": -output["face_angle"] * 25.0,
            "head_x": output.get("head_x", 0.0) * 24.0,
            "head_y": output.get("head_y", 0.0) * 24.0,
        }

    @staticmethod
    def _draw_centered(target, image, center):
        target.blit(image, image.get_rect(center=center))

    @staticmethod
    def _scale_open(image, openness, minimum_ratio=0.06):
        ratio = clamp(openness, minimum_ratio, 1.0)
        new_height = max(2, int(image.get_height() * ratio))
        return pygame.transform.scale(image, (image.get_width(), new_height))

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        p = self.parameters

        self._draw_centered(screen, self.body, self.base_positions["body"])

        self.head_surface.fill((0, 0, 0, 0))
        lc = self.local_center

        def local_pos(pos):
            return (
                lc[0] + pos[0] - self.screen_center[0],
                lc[1] + pos[1] - self.screen_center[1],
            )

        self._draw_centered(self.head_surface, self.face, lc)

        left_eye = self._scale_open(self.left_eye, self.eye_open("left"))
        right_eye = self._scale_open(self.right_eye, self.eye_open("right"))
        self._draw_centered(
            self.head_surface, left_eye,
            local_pos((
                self.base_positions["left_eye"][0] + p["left_eye_x"],
                self.base_positions["left_eye"][1] + p["left_eye_y"],
            )),
        )
        self._draw_centered(
            self.head_surface, right_eye,
            local_pos((
                self.base_positions["right_eye"][0] + p["right_eye_x"],
                self.base_positions["right_eye"][1] + p["right_eye_y"],
            )),
        )

        mouth = self._scale_open(self.mouth, p["mouth_open"], 0.1)
        self._draw_centered(
            self.head_surface, mouth,
            local_pos((
                self.base_positions["mouth"][0] + p["mouth_x"],
                self.base_positions["mouth"][1],
            )),
        )

        self._draw_centered(
            self.head_surface, self.left_brow,
            local_pos((
                self.base_positions["left_brow"][0],
                self.base_positions["left_brow"][1] + p["left_brow_y"],
            )),
        )
        self._draw_centered(
            self.head_surface, self.right_brow,
            local_pos((
                self.base_positions["right_brow"][0],
                self.base_positions["right_brow"][1] + p["right_brow_y"],
            )),
        )

        self._draw_centered(self.head_surface, self.hair_front, lc)

        rotated = pygame.transform.rotate(self.head_surface, p["face_angle"])
        self._draw_centered(screen, rotated, (
            self.screen_center[0] + p["head_x"],
            self.screen_center[1] + p["head_y"],
        ))


# ==============================================================================
# 3. ПРОЦЕДУРНЫЙ СМАЙЛИК
# ==============================================================================

class PixelSmiley(BaseAvatar):
    """Смайлик на сетке 96x96, управляемый всеми выходами модели."""

    NAME = "Pixel Smiley (96x96)"

    def __init__(self, screen_size):
        super().__init__(screen_size)
        self.screen_center = (screen_size[0] // 2, 285)
        self.art_surface = pygame.Surface((96, 96), pygame.SRCALPHA)

    def draw(self, screen):
        screen.fill((30, 36, 53))
        art = self.art_surface
        art.fill((0, 0, 0, 0))
        ink = (65, 40, 32)

        pygame.draw.circle(art, ink, (48, 48), 40)
        pygame.draw.circle(art, (190, 106, 37), (48, 48), 37)
        pygame.draw.circle(art, (239, 163, 44), (47, 45), 35)
        pygame.draw.circle(art, (255, 205, 66), (45, 42), 32)
        pygame.draw.circle(art, (255, 220, 95), (43, 39), 27)
        pygame.draw.rect(art, (255, 240, 157), (26, 23, 9, 3))
        pygame.draw.rect(art, (255, 240, 157), (22, 27, 4, 7))

        for x, y in [(72, 57), (69, 64), (64, 69), (58, 73)]:
            pygame.draw.rect(art, (224, 141, 36), (x, y, 2, 2))

        p = self.parameters
        for side, base_x in [("left", 33), ("right", 63)]:
            x = base_x + round(p[side + "_eye_x"] / 4)
            y = 42 + round(p[side + "_eye_y"] / 4)
            h = max(2, round(17 * self.eye_open(side)))
            rect = pygame.Rect(x - 6, y - h // 2, 12, h)
            pygame.draw.rect(art, ink, rect, border_radius=min(4, h // 2))
            if h >= 7:
                pygame.draw.rect(art, (255, 249, 215), (x - 3, y - h // 2 + 2, 3, 4))
                pygame.draw.rect(art, (123, 78, 45), (x + 2, y + h // 2 - 4, 2, 2))
            by = 27 + round(p[side + "_brow_y"] / 4)
            pygame.draw.lines(
                art, ink, False,
                [(base_x - 8, by + 2), (base_x - 4, by), (base_x + 5, by), (base_x + 8, by + 1)],
                3,
            )
            pygame.draw.rect(art, (244, 146, 65), (base_x - 7, 55, 14, 3))
            pygame.draw.rect(art, (255, 172, 86), (base_x - 4, 54, 8, 2))

        mx = 48 + round(p["mouth_x"] / 4)
        opening = clamp(p["mouth_open"], 0, 1)
        if opening < 0.18:
            pygame.draw.lines(
                art, ink, False,
                [(mx - 11, 64), (mx - 7, 68), (mx + 6, 68), (mx + 11, 64)], 2
            )
        else:
            height = max(6, round(22 * opening))
            pygame.draw.rect(art, ink, (mx - 12, 62, 24, height), border_radius=5)
            pygame.draw.rect(art, (255, 245, 204), (mx - 8, 63, 16, 3))
            if height >= 9:
                pygame.draw.rect(art, (225, 95, 103), (mx - 7, 62 + height - 5, 14, 3))
                pygame.draw.rect(art, (255, 145, 145), (mx - 4, 62 + height - 5, 7, 1))

        rotated = pygame.transform.rotate(art, p["face_angle"])
        scaled = pygame.transform.scale(
            rotated, (rotated.get_width() * 4, rotated.get_height() * 4)
        )

        pygame.draw.ellipse(
            screen, (22, 27, 41), (self.screen_center[0] - 120, 458, 240, 20)
        )
        head_center = (
            self.screen_center[0] + round(p["head_x"] * 60),
            self.screen_center[1] + round(p["head_y"] * 60),
        )
        screen.blit(scaled, scaled.get_rect(center=head_center))


# ==============================================================================
# ИНТЕРФЕЙС
# ==============================================================================

def draw_text_interface(
    screen, font, small_font, text, cursor_pos, status, output,
    avatar_name, backend, thinking, fringe_name,
):
    panel = pygame.Surface((SCREEN_WIDTH, 175), pygame.SRCALPHA)
    panel.fill((16, 22, 32, 230))
    screen.blit(panel, (0, SCREEN_HEIGHT - 175))

    header_text = f"Аватар: {avatar_name} · {backend} · чёлка: {fringe_name}"
    header_surf = small_font.render(header_text, True, (135, 175, 220))
    screen.blit(header_surf, (24, SCREEN_HEIGHT - 165))

    hint_text = "Tab / 1-3: аватар · F2: нейтраль · H: чёлка · Esc: выход"
    hint_surf = small_font.render(hint_text, True, (110, 130, 155))
    screen.blit(hint_surf, (SCREEN_WIDTH - hint_surf.get_width() - 24, SCREEN_HEIGHT - 165))

    hint = font.render("Введите команду / реплику и нажмите Enter:", True, (230, 238, 248))
    screen.blit(hint, (24, SCREEN_HEIGHT - 142))

    input_rect = pygame.Rect(24, SCREEN_HEIGHT - 112, SCREEN_WIDTH - 48, 38)
    pygame.draw.rect(screen, (246, 248, 252), input_rect, border_radius=6)
    pygame.draw.rect(screen, (70, 150, 240), input_rect, 2, border_radius=6)

    left_part = text[:cursor_pos]
    text_surf = font.render(text, True, (20, 28, 38))
    screen.blit(text_surf, (input_rect.x + 12, input_rect.y + 7))

    cursor_x = input_rect.x + 12 + font.size(left_part)[0]
    if int(time.time() * 2.5) % 2 == 0:
        pygame.draw.line(
            screen, (20, 30, 50), (cursor_x, input_rect.y + 7), (cursor_x, input_rect.y + 31), 2
        )

    if thinking:
        dots = "." * (1 + int(time.time() * 3) % 3)
        status = f"Модель думает{dots} {status}"
    status_color = (240, 110, 110) if ("Ошибка" in status or "не ответила" in status) else (
        (255, 200, 120) if thinking else (165, 200, 235)
    )
    status_surf = small_font.render(status, True, status_color)
    screen.blit(status_surf, (24, SCREEN_HEIGHT - 66))

    if output:
        summary = (
            f"Выход: бровь Y: {output['left_brow_y']:+.2f} | "
            f"глаз X/Y: {output['left_eye_x']:+.2f}/{output['left_eye_y']:+.2f} | "
            f"веки: {output['left_eye_open']:.2f} | "
            f"рот: {output['mouth_open']:.2f} | "
            f"угол: {output['face_angle']:+.2f}"
        )
        summary_surf = small_font.render(summary, True, (205, 215, 225))
        screen.blit(summary_surf, (24, SCREEN_HEIGHT - 38))


# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

def main():
    pygame.init()
    pygame.key.set_repeat(350, 35)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Nori — 2D Аватар")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("dejavusans", 20)
    small_font = pygame.font.SysFont("dejavusans", 14)

    nori = NoriAvatar(screen.get_size())
    avatars = [nori, ClassicAvatar(screen.get_size()), PixelSmiley(screen.get_size())]
    current_avatar_idx = 0

    brain = AvatarBrain(QWEN_CACHE_PATH)

    input_text = ""
    cursor_pos = 0
    history = []
    history_idx = -1
    last_output = None
    status = "Введите реплику — решение принимает модель."

    def apply_result(result):
        nonlocal last_output, status
        params = result["params"]
        if params is None:
            params = dict(last_output) if last_output else dict(expressions.NEUTRAL)
        for avatar in avatars:
            avatar.apply_model_output(params, result["seqs"])
        if not result["seqs"]:
            last_output = params
        sequence_info = f" ▶ {', '.join(result['seqs'])}" if result["seqs"] else ""
        status = f"«{result['text']}» → {result['desc']}{sequence_info}  [{result['source']}]"

    pygame.key.start_text_input()
    running = True

    while running:
        delta_time = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.TEXTINPUT:
                input_text = input_text[:cursor_pos] + event.text + input_text[cursor_pos:]
                cursor_pos += len(event.text)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    current_avatar_idx = (current_avatar_idx + 1) % len(avatars)
                    if last_output:
                        avatars[current_avatar_idx].apply_model_output(last_output)

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    idx = event.key - pygame.K_1
                    if idx < len(avatars):
                        current_avatar_idx = idx
                        if last_output:
                            avatars[current_avatar_idx].apply_model_output(last_output)

                elif event.key == pygame.K_h:
                    name = nori.cycle_fringe()
                    status = f"Чёлка: {name}"

                elif event.key == pygame.K_F2:
                    for avatar in avatars:
                        avatar.reset()
                    last_output = None
                    status = "Нейтральная поза"

                elif event.key == pygame.K_BACKSPACE:
                    if cursor_pos > 0:
                        input_text = input_text[:cursor_pos - 1] + input_text[cursor_pos:]
                        cursor_pos -= 1

                elif event.key == pygame.K_DELETE:
                    if cursor_pos < len(input_text):
                        input_text = input_text[:cursor_pos] + input_text[cursor_pos + 1:]

                elif event.key == pygame.K_LEFT:
                    cursor_pos = max(0, cursor_pos - 1)

                elif event.key == pygame.K_RIGHT:
                    cursor_pos = min(len(input_text), cursor_pos + 1)

                elif event.key == pygame.K_HOME:
                    cursor_pos = 0

                elif event.key == pygame.K_END:
                    cursor_pos = len(input_text)

                elif event.key == pygame.K_UP:
                    if history and history_idx < len(history) - 1:
                        history_idx += 1
                        input_text = history[-(history_idx + 1)]
                        cursor_pos = len(input_text)

                elif event.key == pygame.K_DOWN:
                    if history_idx > 0:
                        history_idx -= 1
                        input_text = history[-(history_idx + 1)]
                        cursor_pos = len(input_text)
                    elif history_idx == 0:
                        history_idx = -1
                        input_text = ""
                        cursor_pos = 0

                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    phrase = input_text.strip()
                    if phrase:
                        if not history or history[-1] != phrase:
                            history.append(phrase)
                        history_idx = -1
                        input_text = ""
                        cursor_pos = 0

                        result = brain.ask(phrase)
                        if result is not None:
                            apply_result(result)
                        else:
                            status = f"«{phrase}» — Qwen обдумывает…"

        result = brain.poll()
        if result is not None:
            apply_result(result)

        active_avatar = avatars[current_avatar_idx]
        active_avatar.update(delta_time)
        active_avatar.draw(screen)

        draw_text_interface(
            screen, font, small_font, input_text, cursor_pos, status, last_output,
            active_avatar.NAME, brain.backend_label, brain.busy,
            NoriAvatar.FRINGE_MODES[nori.fringe_mode],
        )

        pygame.display.flip()

    pygame.key.stop_text_input()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    from main_nori_gl import main as gpu_main

    gpu_main()
