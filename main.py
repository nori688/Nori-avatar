"""Главный модуль управления 2D-аватаром с помощью текстовых команд на русском языке.

Поддерживает три вида аватаров:
1. NoriAvatar (Live2D / PSD многослойный аватар высокого разрешения из py.psd) — по умолчанию
2. ClassicAvatar (Пиксельный аниме-арт)
3. PixelSmiley (Процедурный смайлик 96x96)

Переключение аватаров: клавишами Tab, 1, 2, 3 или F1.
Сброс позы: F2.
Выход: Esc.
"""

import json
import math
from pathlib import Path
import sys
import time
import warnings

# Подавляем предупреждения о несовпадении минорных версий scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

import pygame

try:
    import joblib
except ModuleNotFoundError:
    joblib = None

import fallbacks


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
BACKGROUND_COLOR = (40, 48, 64)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "avatar_text_model.joblib"
NORI_MANIFEST_PATH = PROJECT_DIR / "assets" / "nori" / "manifest.json"

MODEL_OUTPUT_NAMES = [
    "left_brow_y",
    "right_brow_y",
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "left_eye_open",
    "right_eye_open",
    "mouth_x",
    "mouth_open",
    "face_angle",
]

ZERO_TO_ONE_OUTPUTS = {
    "left_eye_open",
    "right_eye_open",
    "mouth_open",
}


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
# ML-МОДЕЛЬ
# ==============================================================================

class TextAvatarModel:
    """Преобразует русскую текстовую фразу в нормализованные параметры мимики."""

    def __init__(self, model_path):
        self.pipeline = None
        self.error = ""

        if joblib is None:
            self.error = "Не установлен пакет joblib / scikit-learn"
            return

        if not model_path.exists():
            self.error = f"Файл модели не найден: {model_path}"
            return

        try:
            self.pipeline = joblib.load(str(model_path))
        except Exception as error:
            self.error = f"Ошибка загрузки модели: {error}"

    @property
    def is_ready(self):
        return self.pipeline is not None

    def predict(self, text):
        if not self.is_ready:
            raise RuntimeError(self.error or "Модель не готова")

        raw_prediction = self.pipeline.predict([text])[0]
        result = {}

        for name, raw_value in zip(MODEL_OUTPUT_NAMES, raw_prediction):
            value = float(raw_value)
            if name in ZERO_TO_ONE_OUTPUTS:
                result[name] = clamp(value, 0.0, 1.0)
            else:
                result[name] = clamp(value, -1.0, 1.0)

        return result


# ==============================================================================
# БАЗОВЫЙ КЛАСС АВАТАРА
# ==============================================================================

class BaseAvatar:
    """Базовый класс с независимой от фреймрейта интерполяцией и микро-анимациями."""

    NAME = "Base"

    def __init__(self, screen_size):
        self.screen_size = screen_size
        self.time = 0.0
        self.blink_timer = 3.0
        self.is_blinking = False
        self.blink_progress = 0.0

        neutral = {
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
        }
        self.parameters = neutral.copy()
        self.target_parameters = neutral.copy()
        self.raw_target = neutral.copy()

    def apply_model_output(self, output):
        self.raw_target = output.copy()
        self.map_model_parameters(output)

    def map_model_parameters(self, output):
        """Переопределяется в подклассах для пересчёта нормализованных выходов."""
        self.target_parameters = output.copy()

    def reset(self):
        neutral_output = {
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
        }
        self.apply_model_output(neutral_output)

    def update(self, delta_time):
        self.time += delta_time

        # Естественное моргание в покое (раз в 3.5–5 секунд)
        self.blink_timer -= delta_time
        if self.blink_timer <= 0.0:
            self.is_blinking = True
            self.blink_progress = 0.0
            self.blink_timer = 3.5 + (hash(str(self.time)) % 200) / 100.0

        blink_factor = 1.0
        if self.is_blinking:
            self.blink_progress += delta_time / 0.16
            if self.blink_progress >= 1.0:
                self.is_blinking = False
            else:
                # Параболическое прикрытие века
                blink_factor = 4.0 * (self.blink_progress - 0.5) ** 2

        # Экспоненциальное сглаживание, независимое от падений FPS
        smoothing_factor = 1.0 - math.exp(-6.5 * delta_time)
        for name in self.parameters:
            target = self.target_parameters[name]
            if "eye_open" in name and self.is_blinking:
                target = min(target, target * blink_factor)
            current = self.parameters[name]
            self.parameters[name] = current + (target - current) * smoothing_factor

    def draw(self, screen):
        raise NotImplementedError


# ==============================================================================
# 1. LIVE2D / PSD АВАТАР (NORI)
# ==============================================================================

class NoriAvatar(BaseAvatar):
    """Полноценный Live2D-аватар на основе экспортированных слоёв py.psd."""

    NAME = "Nori (Live2D PSD)"

    def __init__(self, screen_size):
        super().__init__(screen_size)
        self.screen_head = (screen_size[0] // 2, 235)
        self.scale = 0.35

        # Загрузка манифеста и слоев
        self.layers_info = {}
        self.surfaces = {}

        if NORI_MANIFEST_PATH.exists():
            with open(NORI_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.layers_info = manifest.get("layers", {})
            for name, info in self.layers_info.items():
                p = PROJECT_DIR / "assets" / "nori" / info["file"]
                if p.exists():
                    raw = pygame.image.load(str(p)).convert_alpha()
                    w = max(1, round(raw.get_width() * self.scale))
                    h = max(1, round(raw.get_height() * self.scale))
                    self.surfaces[name] = pygame.transform.smoothscale(raw, (w, h))

        # Переиспользуемые холсты для исключения лишних аллокаций памяти
        self.head_canvas = pygame.Surface((640, 640), pygame.SRCALPHA)
        self.head_center = (320, 320)

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
            "face_angle": -output["face_angle"] * 22.0,
        }

    def _blit_layer(self, dest, name, center, offset=(0, 0), scale_y=1.0):
        if name not in self.surfaces or name not in self.layers_info:
            return
        info = self.layers_info[name]
        rc = info["rel_center"]
        x = center[0] + round(rc[0] * self.scale) + offset[0]
        y = center[1] + round(rc[1] * self.scale) + offset[1]

        surf = self.surfaces[name]
        if abs(scale_y - 1.0) > 0.05:
            new_h = max(2, int(surf.get_height() * scale_y))
            surf = pygame.transform.scale(surf, (surf.get_width(), new_h))

        rect = surf.get_rect(center=(x, y))
        dest.blit(surf, rect)

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        p = self.parameters

        # Микро-анимация дыхания
        breath_y = math.sin(self.time * 2.2) * 2.5
        hair_sway = math.cos(self.time * 1.8) * 1.5

        # 1. Задние волосы и тело остаются на основном экране
        body_center = (self.screen_head[0], self.screen_head[1] + breath_y * 0.5)
        self._blit_layer(screen, "back_hair", body_center, offset=(0, -2))
        self._blit_layer(screen, "body", body_center, offset=(0, round(breath_y)))
        self._blit_layer(screen, "neck", body_center, offset=(0, round(breath_y * 0.7)))

        # 2. Отрисовка головы на локальном прозрачном холсте для вращения
        self.head_canvas.fill((0, 0, 0, 0))
        hc = self.head_center

        # База лица
        self._blit_layer(self.head_canvas, "face_base", hc)

        # Белки глаз
        self._blit_layer(self.head_canvas, "left_eye_white", hc)
        self._blit_layer(self.head_canvas, "right_eye_white", hc)

        # Зрачки с перемещением взгляда
        left_eye_offset = (round(p["left_eye_x"]), round(p["left_eye_y"]))
        right_eye_offset = (round(p["right_eye_x"]), round(p["right_eye_y"]))
        self._blit_layer(
            self.head_canvas, "left_eye_iris", hc, offset=left_eye_offset,
            scale_y=clamp(p["left_eye_open"], 0.2, 1.0)
        )
        self._blit_layer(
            self.head_canvas, "right_eye_iris", hc, offset=right_eye_offset,
            scale_y=clamp(p["right_eye_open"], 0.2, 1.0)
        )

        # Веки (при закрытии верхнее веко опускается)
        left_lid_offset = (0, round((1.0 - p["left_eye_open"]) * 16))
        right_lid_offset = (0, round((1.0 - p["right_eye_open"]) * 16))
        self._blit_layer(self.head_canvas, "left_eyelid_lower", hc)
        self._blit_layer(self.head_canvas, "right_eyelid_lower", hc)
        self._blit_layer(self.head_canvas, "left_eyelid_upper", hc, offset=left_lid_offset)
        self._blit_layer(self.head_canvas, "right_eyelid_upper", hc, offset=right_lid_offset)

        # Нос
        self._blit_layer(self.head_canvas, "nose", hc)

        # Рот с масштабированием открытия
        mouth_offset = (round(p["mouth_x"]), 0)
        mouth_ratio = clamp(p["mouth_open"] * 1.5, 0.15, 1.4)
        self._blit_layer(self.head_canvas, "mouth", hc, offset=mouth_offset, scale_y=mouth_ratio)

        # Брови
        self._blit_layer(self.head_canvas, "left_brow", hc, offset=(0, round(p["left_brow_y"])))
        self._blit_layer(self.head_canvas, "right_brow", hc, offset=(0, round(p["right_brow_y"])))

        # Передняя челка
        self._blit_layer(self.head_canvas, "front_hair", hc, offset=(round(hair_sway), 0))

        # 3. Вращение всей головы единым узлом
        rotated_head = pygame.transform.rotate(self.head_canvas, p["face_angle"])
        head_screen_pos = (self.screen_head[0], self.screen_head[1] + breath_y)
        rect = rotated_head.get_rect(center=head_screen_pos)
        screen.blit(rotated_head, rect)


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
        brow = load_layer(
            "brow.png", (60, 12), fallbacks.draw_reference_brow_fallback
        )
        eye = load_layer(
            "eye.png", (72, 40), fallbacks.draw_reference_eye_fallback
        )
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

        # Тело
        self._draw_centered(screen, self.body, self.base_positions["body"])

        # Сборка головы
        self.head_surface.fill((0, 0, 0, 0))
        lc = self.local_center

        def local_pos(pos):
            return (
                lc[0] + pos[0] - self.screen_center[0],
                lc[1] + pos[1] - self.screen_center[1],
            )

        self._draw_centered(self.head_surface, self.face, lc)

        left_eye = self._scale_open(self.left_eye, p["left_eye_open"])
        right_eye = self._scale_open(self.right_eye, p["right_eye_open"])
        self._draw_centered(
            self.head_surface, left_eye,
            local_pos((self.base_positions["left_eye"][0] + p["left_eye_x"],
                       self.base_positions["left_eye"][1] + p["left_eye_y"]))
        )
        self._draw_centered(
            self.head_surface, right_eye,
            local_pos((self.base_positions["right_eye"][0] + p["right_eye_x"],
                       self.base_positions["right_eye"][1] + p["right_eye_y"]))
        )

        mouth = self._scale_open(self.mouth, p["mouth_open"], 0.1)
        self._draw_centered(
            self.head_surface, mouth,
            local_pos((self.base_positions["mouth"][0] + p["mouth_x"],
                       self.base_positions["mouth"][1]))
        )

        self._draw_centered(
            self.head_surface, self.left_brow,
            local_pos((self.base_positions["left_brow"][0],
                       self.base_positions["left_brow"][1] + p["left_brow_y"]))
        )
        self._draw_centered(
            self.head_surface, self.right_brow,
            local_pos((self.base_positions["right_brow"][0],
                       self.base_positions["right_brow"][1] + p["right_brow_y"]))
        )

        self._draw_centered(self.head_surface, self.hair_front, lc)

        rotated = pygame.transform.rotate(self.head_surface, p["face_angle"])
        self._draw_centered(screen, rotated, self.screen_center)


# ==============================================================================
# 3. ПРОЦЕДУРНЫЙ СМАЙЛИК
# ==============================================================================

class PixelSmiley(BaseAvatar):
    """Смайлик на сетке 96x96, управляемый всеми 11 выходами."""

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
            h = max(2, round(17 * p[side + "_eye_open"]))
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
        scaled = pygame.transform.scale(rotated, (rotated.get_width() * 4, rotated.get_height() * 4))

        pygame.draw.ellipse(
            screen, (22, 27, 41),
            (self.screen_center[0] - 120, 458, 240, 20)
        )
        screen.blit(scaled, scaled.get_rect(center=self.screen_center))


# ==============================================================================
# ИНТЕРФЕЙС
# ==============================================================================

def draw_text_interface(screen, font, small_font, text, cursor_pos, status, output, avatar_name):
    """Рисует панель ввода, курсор и статус."""
    panel = pygame.Surface((SCREEN_WIDTH, 175), pygame.SRCALPHA)
    panel.fill((16, 22, 32, 230))
    screen.blit(panel, (0, SCREEN_HEIGHT - 175))

    # Верхняя строчка: аватар и подсказки горячих клавиш
    header_text = f"Аватар: {avatar_name}  [Tab: сменить | F2: нейтраль | Esc: выход]"
    header_surf = small_font.render(header_text, True, (135, 175, 220))
    screen.blit(header_surf, (24, SCREEN_HEIGHT - 165))

    hint = font.render("Введите команду / реплику и нажмите Enter:", True, (230, 238, 248))
    screen.blit(hint, (24, SCREEN_HEIGHT - 142))

    # Поле ввода
    input_rect = pygame.Rect(24, SCREEN_HEIGHT - 112, SCREEN_WIDTH - 48, 38)
    pygame.draw.rect(screen, (246, 248, 252), input_rect, border_radius=6)
    pygame.draw.rect(screen, (70, 150, 240), input_rect, 2, border_radius=6)

    # Отрисовка текста и каретки курсора
    left_part = text[:cursor_pos]
    text_surf = font.render(text, True, (20, 28, 38))
    screen.blit(text_surf, (input_rect.x + 12, input_rect.y + 7))

    # Мигающий курсор
    cursor_x = input_rect.x + 12 + font.size(left_part)[0]
    if int(time.time() * 2.5) % 2 == 0:
        pygame.draw.line(screen, (20, 30, 50), (cursor_x, input_rect.y + 7),
                         (cursor_x, input_rect.y + 31), 2)

    # Статус
    status_color = (240, 110, 110) if "Ошибка" in status else (165, 200, 235)
    status_surf = small_font.render(status, True, status_color)
    screen.blit(status_surf, (24, SCREEN_HEIGHT - 66))

    # Метрики предсказания
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
    pygame.display.set_caption("Nori — 2D Аватар с текстовым управлением")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("dejavusans", 20)
    small_font = pygame.font.SysFont("dejavusans", 14)

    # Инициализация аватаров
    avatars = [
        NoriAvatar(screen.get_size()),
        ClassicAvatar(screen.get_size()),
        PixelSmiley(screen.get_size()),
    ]
    current_avatar_idx = 0

    model = TextAvatarModel(MODEL_PATH)
    input_text = ""
    cursor_pos = 0
    history = []
    history_idx = -1
    last_output = None

    status = "Модель готова. Введите реплику (например: 'улыбнись', 'посмотри налево')."
    if not model.is_ready:
        status = model.error

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

                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    idx = event.key - pygame.K_1
                    if idx < len(avatars):
                        current_avatar_idx = idx
                        if last_output:
                            avatars[current_avatar_idx].apply_model_output(last_output)

                elif event.key == pygame.K_F2:
                    for a in avatars:
                        a.reset()
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

                        if model.is_ready:
                            try:
                                last_output = model.predict(phrase)
                                for a in avatars:
                                    a.apply_model_output(last_output)
                                status = f"Вход: «{phrase}»"
                            except Exception as error:
                                status = f"Ошибка предсказания: {error}"

        active_avatar = avatars[current_avatar_idx]
        active_avatar.update(delta_time)
        active_avatar.draw(screen)

        draw_text_interface(
            screen, font, small_font, input_text, cursor_pos, status, last_output,
            active_avatar.NAME
        )

        pygame.display.flip()

    pygame.key.stop_text_input()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
