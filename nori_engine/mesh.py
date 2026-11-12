"""Меш-деформация спрайтов: одна картинка превращается в любую форму.

Сетка контрольных точек накладывается на исходный спрайт; для каждого
кадра считается новое положение точек, и спрайт перерисовывается через
триангуляцию с обратным аффинным преобразованием и билинейной выборкой.

Размер имеет значение: для мелких деталей (рот, глаз, бровь) это
доли миллисекунды, для целой головы — уже дорого. Поэтому голову и тело
по-прежнему двигаем трансформациями, а лицо гнём мешем.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pygame


def grid_points(cols: int, rows: int, rect: Tuple[float, float, float, float]) -> np.ndarray:
    """Равномерная сетка точек внутри прямоугольника (x, y, w, h)."""
    x, y, w, h = rect
    xs = np.linspace(x, x + w, cols, dtype=np.float32)
    ys = np.linspace(y, y + h, rows, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def _surface_to_array(surface: pygame.Surface) -> np.ndarray:
    """RGBA-массив (w, h, 4) float32 с прямым альфа-каналом."""
    rgb = pygame.surfarray.pixels3d(surface).astype(np.float32)
    alpha = pygame.surfarray.pixels_alpha(surface).astype(np.float32) / 255.0
    return np.dstack([rgb, alpha[:, :, None]])


def _array_to_surface(array: np.ndarray) -> pygame.Surface:
    """Обратное преобразование: (w, h, 4) -> Surface с альфой."""
    data = np.clip(array, 0, 255).astype(np.uint8)
    data[:, :, 3] = np.clip(array[:, :, 3] * 255.0, 0, 255).astype(np.uint8)
    # pygame ждёт построчный порядок (y, x), а массивы у нас (x, y)
    return pygame.image.frombuffer(data.transpose(1, 0, 2).tobytes(), data.shape[:2], "RGBA")


def _sample_bilinear(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Билинейная выборка из (w, h, 4)."""
    w, h = src.shape[0], src.shape[1]
    xs = np.clip(xs, 0, w - 1.001)
    ys = np.clip(ys, 0, h - 1.001)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (xs - x0)[:, None]
    wy = (ys - y0)[:, None]

    top = src[x0, y0] * (1 - wx) + src[x1, y0] * wx
    bottom = src[x0, y1] * (1 - wx) + src[x1, y1] * wx
    return top * (1 - wy) + bottom * wy


def _axis_weights(coords: np.ndarray, size: int):
    """Веса билинейной интерполяции вдоль одной оси для каждого пикселя."""
    pixels = np.arange(size, dtype=np.float32)
    idx = np.clip(np.searchsorted(coords, pixels) - 1, 0, len(coords) - 2)
    x0 = coords[idx]
    x1 = coords[idx + 1]
    t = np.clip((pixels - x0) / np.maximum(x1 - x0, 1e-6), 0.0, 1.0)
    return t.astype(np.float32), idx.astype(np.int32), (idx + 1).astype(np.int32)


def _sample_field(field: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Ближайшая выборка из поля (w, h, 2) — дёшево и достаточно."""
    w, h = field.shape[0], field.shape[1]
    xi = np.clip(xs, 0, w - 1).astype(np.int32)
    yi = np.clip(ys, 0, h - 1).astype(np.int32)
    return field[xi, yi][..., 0], field[xi, yi][..., 1]


class Mesh:
    """Сетка контрольных точек, наложенная на спрайт."""

    def __init__(self, surface: pygame.Surface, cols: int = 5, rows: int = 5):
        self.surface = surface
        self.array = _surface_to_array(surface)
        self.dirty = True
        self.cols = cols
        self.rows = rows
        self.base = grid_points(cols, rows, (0, 0, surface.get_width(), surface.get_height()))
        self.points = self.base.copy()

    # ------------------------------------------------------------------
    def reset(self):
        self.points = self.base.copy()

    def translate(self, dx: float, dy: float):
        self.points[:, :, 0] += dx
        self.points[:, :, 1] += dy

    def deform(self, fn):
        """fn(u, v, x, y) -> (x, y), где u,v — нормализованные координаты -1..1."""
        cols, rows = self.cols, self.rows
        for r in range(rows):
            for c in range(cols):
                u = (c / (cols - 1) - 0.5) * 2.0
                v = (r / (rows - 1) - 0.5) * 2.0
                x, y = self.points[r, c]
                self.points[r, c] = fn(u, v, x, y)

    # ------------------------------------------------------------------
    def render(self) -> Tuple[pygame.Surface, Tuple[int, int]]:
        """Быстрый рендер: поле смещений на весь спрайт, один проход numpy.

        Треугольный вариант (render_triangles) точнее, но на Python дорогой;
        для плавных деформаций лица разница не видна, а скорость выше в разы.
        """
        src = self.array
        width, height = src.shape[0], src.shape[1]
        field = self._displacement_field()

        pts = self.points
        min_x = int(np.floor(pts[:, :, 0].min()))
        min_y = int(np.floor(pts[:, :, 1].min()))
        max_x = int(np.ceil(pts[:, :, 0].max()))
        max_y = int(np.ceil(pts[:, :, 1].max()))
        out_w = max(1, max_x - min_x)
        out_h = max(1, max_y - min_y)

        gx, gy = np.meshgrid(
            np.arange(min_x, max_x, dtype=np.float32),
            np.arange(min_y, max_y, dtype=np.float32),
            indexing="ij",
        )
        px = gx
        py = gy

        # Обратная задача: подбираем источник, из которого пиксель пришёл
        sx = px.copy()
        sy = py.copy()
        for _ in range(6):
            dx, dy = _sample_field(field, sx, sy)
            sx = px - dx
            sy = py - dy

        inside = (
            (sx >= 0) & (sx <= width - 1.001) & (sy >= 0) & (sy <= height - 1.001)
        )
        if not inside.any():
            empty = np.zeros((out_w, out_h, 4), dtype=np.float32)
            return _array_to_surface(empty), (min_x, min_y)

        sampled = _sample_bilinear(src, sx[inside], sy[inside])
        out = np.zeros((out_w, out_h, 4), dtype=np.float32)
        region = out[inside]
        alpha = sampled[:, 3:4]
        region[:, :3] = sampled[:, :3]
        region[:, 3:4] = alpha
        # В пределах спрайта альфа может быть <1 (сглаженные края) — так и надо
        out[inside] = region
        return _array_to_surface(out), (min_x, min_y)

    def _displacement_field(self) -> np.ndarray:
        """Плотное поле смещений (w, h, 2), билинейно из контрольных точек."""
        disp = (self.points - self.base).astype(np.float32)
        height, width = self.array.shape[1], self.array.shape[0]

        fx, cx0, cx1 = _axis_weights(self.base[0, :, 0], width)
        fy, ry0, ry1 = _axis_weights(self.base[:, 0, 1], height)

        top = disp[ry0][:, cx0] * (1 - fx)[None, :, None] + disp[ry0][:, cx1] * fx[None, :, None]
        bottom = disp[ry1][:, cx0] * (1 - fx)[None, :, None] + disp[ry1][:, cx1] * fx[None, :, None]
        dense = top * (1 - fy)[:, None, None] + bottom * fy[:, None, None]
        return dense.transpose(1, 0, 2).astype(np.float32)

    def render_triangles(self) -> Tuple[pygame.Surface, Tuple[int, int]]:
        """Возвращает поверхность и смещение её левого верхнего угла."""
        src = self.array
        dst = self.points

        min_x = float(np.floor(dst[:, :, 0].min()))
        min_y = float(np.floor(dst[:, :, 1].min()))
        max_x = float(np.ceil(dst[:, :, 0].max()))
        max_y = float(np.ceil(dst[:, :, 1].max()))
        out_w = max(1, int(max_x - min_x))
        out_h = max(1, int(max_y - min_y))

        out = np.zeros((out_w, out_h, 4), dtype=np.float32)
        local = dst - np.array([min_x, min_y], dtype=np.float32)

        cols, rows = self.cols, self.rows
        for r in range(rows - 1):
            for c in range(cols - 1):
                s00 = self.base[r, c]
                s10 = self.base[r, c + 1]
                s11 = self.base[r + 1, c + 1]
                s01 = self.base[r + 1, c]

                d00 = local[r, c]
                d10 = local[r, c + 1]
                d11 = local[r + 1, c + 1]
                d01 = local[r + 1, c]

                for tri in (
                    ((s00, s10, s11), (d00, d10, d11)),
                    ((s00, s11, s01), (d00, d11, d01)),
                ):
                    self._raster(out, src, tri[0], tri[1])

        return _array_to_surface(out), (int(min_x), int(min_y))

    @staticmethod
    def _raster(out: np.ndarray, src: np.ndarray, sp, dp):
        s = [np.asarray(p, dtype=np.float32) for p in sp]
        d = [np.asarray(p, dtype=np.float32) for p in dp]

        area = (d[1][0] - d[0][0]) * (d[2][1] - d[0][1]) - (d[2][0] - d[0][0]) * (d[1][1] - d[0][1])
        if abs(area) < 1e-6:
            return

        x_lo = max(0, int(math.floor(min(p[0] for p in d))))
        x_hi = min(out.shape[0] - 1, int(math.ceil(max(p[0] for p in d))))
        y_lo = max(0, int(math.floor(min(p[1] for p in d))))
        y_hi = min(out.shape[1] - 1, int(math.ceil(max(p[1] for p in d))))
        if x_hi < x_lo or y_hi < y_lo:
            return

        xs = np.arange(x_lo, x_hi + 1, dtype=np.float32)
        ys = np.arange(y_lo, y_hi + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")

        # Обратное аффинное преобразование: пиксель выхода -> барицентрика
        inv = 1.0 / area
        px = gx - d[0][0]
        py = gy - d[0][1]
        u = ((d[2][1] - d[0][1]) * px - (d[2][0] - d[0][0]) * py) * inv
        v = ((d[1][0] - d[0][0]) * py - (d[1][1] - d[0][1]) * px) * inv
        inside = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1 + 1e-4)
        if not inside.any():
            return

        sx = s[0][0] + u * (s[1][0] - s[0][0]) + v * (s[2][0] - s[0][0])
        sy = s[0][1] + u * (s[1][1] - s[0][1]) + v * (s[2][1] - s[0][1])

        sampled = _sample_bilinear(src, sx[inside], sy[inside])
        if not (sampled[:, 3] > 0.003).any():
            return

        target = out[x_lo:x_hi + 1, y_lo:y_hi + 1]

        # Композитинг только по маске (иначе затрём соседние треугольники)
        sub = target[inside]
        sa = sampled[:, 3:4]
        da = sub[:, 3:4]
        out_a = sa + da * (1 - sa)
        safe = np.where(out_a > 1e-6, out_a, 1.0)
        sub[:, :3] = (sampled[:, :3] * sa + sub[:, :3] * da * (1 - sa)) / safe
        sub[:, 3:4] = out_a
        target[inside] = sub


def warp_surface(surface: pygame.Surface, points: np.ndarray, base: np.ndarray) -> pygame.Surface:
    """Разовый варп: спрайт и две сетки точек (исходная и целевая)."""
    mesh = Mesh(surface, points.shape[1], points.shape[0])
    mesh.base = base.astype(np.float32)
    mesh.points = points.astype(np.float32)
    return mesh.render()[0]
