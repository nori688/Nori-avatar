"""Мини-рантайм Live2D: рисует модель по снятому ригу, без SDK.

Риг — это JSON с вершинами, UV, индексами и сэмплами деформации по
параметрам. Поза считается как «нейтраль + сумма отклонений», затем все
части сливаются в один меш и рисуются одним вызовом на GPU.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Optional

import moderngl
import numpy as np
import pygame

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
uniform vec2 u_scale;
uniform vec2 u_offset;
void main() {
    v_uv = in_uv;
    vec2 p = in_pos * u_scale + u_offset;      // пиксели холста
    vec2 ndc = vec2(p.x / u_offset.z * 2.0 - 1.0, 0.0);
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
"""


class Live2DRig:
    """Модель, восстановленная из экспортированного рига."""

    def __init__(self, rig_path, texture_path, ctx=None, skip_parts=None, only_parts=None):
        rig_path = Path(rig_path)
        opener = gzip.open if rig_path.suffix == ".gz" else open
        with opener(rig_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        self.draws = data["draws"]
        self.base = np.array(data["base"], dtype=np.float32).reshape(-1, 2)
        self.params = {p["id"]: p for p in data["params"]}
        self.values: Dict[str, float] = {pid: 0.0 for pid in self.params}

        # сэмплы в виде массивов (n_verts*2)
        for p in self.params.values():
            p["min_arr"] = (np.array(p["atMin"], dtype=np.float32) - data["base"]).reshape(-1, 2)
            p["max_arr"] = (np.array(p["atMax"], dtype=np.float32) - data["base"]).reshape(-1, 2)

        self.positions = self.base.copy()

        # общий меш: одна текстура, один вызов
        def order_of(i):
            o = self.draws[i]["order"]
            if isinstance(o, (int, float)):
                return o
            return o[str(i)] if isinstance(o, dict) else o[i]

        # part 7 — фоновый меш, 26/27/31 — ВТОРАЯ поза (второй посох и две
        # поднятые руки). В оригинале их прячет pose3.json, ядро отдаёт нам их
        # с прозрачностью 1, поэтому фильтруем по номеру части.
        self.skip_parts = set(skip_parts or ()) | {7, 26, 27, 31}
        visible = [
            i for i in range(len(self.draws))
            if self.draws[i].get("opacity", 1.0) > 0.01
            and self.draws[i].get("part", -1) not in self.skip_parts
            and (only_parts is None or self.draws[i].get("part", -1) in only_parts)
        ]
        order = sorted(visible, key=order_of)

        counts = [len(d["uv"]) // 2 for d in self.draws]
        orig_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        index_list = []
        uv_list = []
        vertex_order = []
        vertex_offset = 0
        for i in order:
            d = self.draws[i]
            idx = np.array(d["idx"], dtype=np.uint32) + vertex_offset
            index_list.append(idx)
            uv_list.append(np.array(d["uv"], dtype=np.float32).reshape(-1, 2))
            vertex_order.append(np.arange(orig_offsets[i], orig_offsets[i + 1]))
            vertex_offset += counts[i]

        self.indices = np.concatenate(index_list)
        self.uv = np.concatenate(uv_list)
        # атлас грузится сверху вниз, а ядро отдаёт V снизу — переворачиваем
        self.uv[:, 1] = 1.0 - self.uv[:, 1]
        self.vertex_order = np.concatenate(vertex_order)
        self.vertex_count = vertex_offset

        self.ctx = ctx or moderngl.create_standalone_context(backend="egl")
        self.texture = self._make_texture(Path(texture_path))
        self.fbo = None
        self.size = None
        self._build_program()

    # ------------------------------------------------------------------
    def _make_texture(self, path: Path):
        surface = pygame.image.load(str(path))
        if surface.get_alpha() is None and not surface.get_flags() & pygame.SRCALPHA:
            surface = surface.convert_alpha() if pygame.display.get_surface() else surface
        data = pygame.image.tostring(surface, "RGBA", False)
        w, h = surface.get_size()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4)).astype(np.float32)
        arr[:, :, :3] *= arr[:, :, 3:4] / 255.0          # premultiplied
        tex = self.ctx.texture((w, h), 4, arr.astype(np.uint8).tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def _build_program(self):
        self.prog = self.ctx.program(
            vertex_shader="""#version 330
            in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
            uniform vec2 u_scale; uniform vec2 u_offset; uniform vec2 u_size;
            void main() {
                v_uv = in_uv;
                vec2 p = in_pos * u_scale + u_offset;
                vec2 ndc = vec2(p.x / u_size.x * 2.0 - 1.0, 1.0 - p.y / u_size.y * 2.0);
                gl_Position = vec4(ndc, 0.0, 1.0);
            }""",
            fragment_shader="""#version 330
            in vec2 v_uv; out vec4 f_color;
            uniform sampler2D u_tex;
            void main() { f_color = texture(u_tex, v_uv); }""",
        )
        self.vbo = self.ctx.buffer(reserve=self.vertex_count * 4 * 4, dynamic=True)
        self.ibo = self.ctx.buffer(self.indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")], self.ibo
        )
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA

    # ------------------------------------------------------------------
    def set_params(self, values: Dict[str, float]):
        pos = self.base.copy()
        for pid, p in self.params.items():
            v = values.get(pid)
            if v is None:
                continue
            span = p["max"] - p["min"]
            t = 0.0 if span == 0 else min(1.0, max(0.0, (v - p["min"]) / span))
            pos += p["min_arr"] * (1.0 - t) + p["max_arr"] * t
        self.positions = pos

    def bounds(self):
        pts = self.positions[self.vertex_order]
        return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())

    # ------------------------------------------------------------------
    def draw_into(self, fbo, width, height, margin=0.98, clear_color=(0.0, 0.0, 0.0, 0.0)):
        """Рисует модель прямо в переданный фреймбуфер (окно), без чтения кадра."""
        if self.size != (width, height):
            self.size = (width, height)
            self.prog["u_size"].value = (float(width), float(height))

        x0, y0, x1, y1 = self.bounds()
        bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        scale = min(width * margin / bw, height * margin / bh)
        offset = ((width - bw * scale) / 2 - x0 * scale,
                  (height - bh * scale) / 2 + y1 * scale)

        data = np.empty((self.vertex_count, 4), dtype=np.float32)
        data[:, 0:2] = self.positions[self.vertex_order]
        data[:, 2:4] = self.uv
        self.vbo.write(data.tobytes())
        self.prog["u_scale"].value = (float(scale), float(-scale))
        self.prog["u_offset"].value = (float(offset[0]), float(offset[1]))

        fbo.use()
        if clear_color is not None:
            fbo.clear(*clear_color)
        self.texture.use(0)
        self.prog["u_tex"].value = 0
        self.vao.render()

    def render(self, width: int, height: int, margin: float = 0.98) -> pygame.Surface:
        if self.fbo is None or self.size != (width, height):
            self.fbo = self.ctx.simple_framebuffer((width, height), components=4)
            self.size = (width, height)
            self.prog["u_size"].value = (float(width), float(height))

        x0, y0, x1, y1 = self.bounds()
        bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        scale = min(width * margin / bw, height * margin / bh)
        # ось Y у ядра направлена вниз — инвертируем при проекции
        offset = np.array([
            (width - bw * scale) / 2 - x0 * scale,
            (height - bh * scale) / 2 + y1 * scale,
        ], dtype=np.float32)

        data = np.empty((self.vertex_count, 4), dtype=np.float32)
        data[:, 0:2] = self.positions[self.vertex_order]
        data[:, 2:4] = self.uv
        self.vbo.write(data.tobytes())

        self.prog["u_scale"].value = (float(scale), float(-scale))
        self.prog["u_offset"].value = tuple(offset)

        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.texture.use(0)
        self.prog["u_tex"].value = 0
        self.vao.render()

        raw = self.fbo.read(components=4, dtype="f1")
        surface = pygame.image.frombuffer(raw, (width, height), "RGBA")
        return pygame.transform.flip(surface, False, True)
