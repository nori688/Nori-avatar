"""GPU-рендер слоёв: та же сетка, но гнётся в вершинном шейдере.

Контекст создаётся через EGL и садится на дискретную карту (на этой машине —
NVIDIA P106M). Кадр рисуется в offscreen-фреймбуфер и забирается одним
чтением, поэтому CPU занимается только логикой и вводом.

Нагрузка на CPU на кадр: обновить 25 вершин на слой (сотни байт) и один
readback буфера.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import moderngl
import numpy as np
import pygame

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
uniform vec2 u_size;
void main() {
    v_uv = in_uv;
    vec2 ndc = vec2(in_pos.x / u_size.x * 2.0 - 1.0,
                    1.0 - in_pos.y / u_size.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_tex;
uniform float u_alpha;
void main() {
    vec4 c = texture(u_tex, v_uv);
    f_color = vec4(c.rgb, c.a * u_alpha);
}
"""


def _surface_bytes(surface: pygame.Surface) -> bytes:
    return pygame.image.tostring(surface, "RGBA", False)


class GPUCanvas:
    """Offscreen-холст на GPU с рисованием деформированных слоёв."""

    def __init__(self, size: Tuple[int, int], ctx: Optional[moderngl.Context] = None):
        self.size = size
        self.ctx = ctx or moderngl.create_standalone_context(backend="egl")
        self.fbo = self.ctx.simple_framebuffer(size, components=4)
        self.fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        self.prog["u_size"].value = size
        self._textures: Dict[str, moderngl.Texture] = {}
        self._buffers: Dict[str, Tuple] = {}

    # ------------------------------------------------------------------
    @property
    def renderer(self) -> str:
        return self.ctx.info.get("GL_RENDERER", "?")

    def upload(self, name: str, surface: pygame.Surface):
        """Загружает спрайт в текстуру (один раз)."""
        if name in self._textures:
            return
        # Текстуры грузим premultiplied: иначе прозрачные пиксели (в PNG они
        # белые) при билинейной фильтрации дают светлую кайму по краям.
        data = _surface_bytes(surface)
        width, height = surface.get_size()
        array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4)).astype(np.float32)
        array[:, :, :3] *= array[:, :, 3:4] / 255.0
        data = array.astype(np.uint8).tobytes()
        texture = self.ctx.texture(surface.get_size(), 4, data)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        self._textures[name] = texture

    def _buffers_for(self, name: str, cols: int, rows: int, uv_rect):
        key = (name, cols, rows, tuple(round(v, 5) for v in uv_rect))
        if key in self._buffers:
            return self._buffers[key]

        u0, v0, u1, v1 = uv_rect
        uvs = []
        indices = []
        for r in range(rows):
            for c in range(cols):
                uvs.append((u0 + (u1 - u0) * c / (cols - 1),
                            v0 + (v1 - v0) * r / (rows - 1)))
        for r in range(rows - 1):
            for c in range(cols - 1):
                i = r * cols + c
                indices += [i, i + 1, i + cols + 1, i, i + cols + 1, i + cols]

        vbo = self.ctx.buffer(reserve=(cols * rows) * 4 * 4)  # x, y, u, v
        ibo = self.ctx.buffer(np.array(indices, dtype="i4").tobytes())
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, "2f 2f", "in_pos", "in_uv")], ibo
        )
        self._buffers[key] = (vbo, ibo, vao, np.array(uvs, dtype="f4"))
        return self._buffers[key]

    # ------------------------------------------------------------------
    def clear(self, color=(40, 48, 64, 255)):
        self.fbo.use()
        self.fbo.clear(*[v / 255.0 for v in color])

    def draw(self, name: str, points: np.ndarray, cols: int, rows: int,
             uv_rect=(0.0, 0.0, 1.0, 1.0), alpha: float = 1.0):
        """Рисует слой по сетке точек (в пикселях холста) с нужным куском текстуры."""
        if name not in self._textures:
            return
        vbo, _ibo, vao, uvs = self._buffers_for(name, cols, rows, uv_rect)

        pts = np.asarray(points, dtype="f4").reshape(rows, cols, 2)
        data = np.empty((rows * cols, 4), dtype="f4")
        data[:, 0] = pts[:, :, 0].reshape(-1)
        data[:, 1] = pts[:, :, 1].reshape(-1)
        data[:, 2:] = uvs
        vbo.write(data.tobytes())

        self._textures[name].use(0)
        self.prog["u_tex"].value = 0
        self.prog["u_alpha"].value = alpha
        vao.render()

    def read(self, viewport=None) -> pygame.Surface:
        """Забирает кадр (или его часть) и переворачивает по Y.

        viewport задаётся в координатах холста сверху-вниз: (x, y, w, h).
        Чтение только нужной области экономит передачу по PCIe.
        """
        if viewport is None:
            data = self.fbo.read(components=4, dtype="f1")
            surface = pygame.image.frombuffer(data, self.size, "RGBA")
            return pygame.transform.flip(surface, False, True)

        x, y, w, h = [int(v) for v in viewport]
        gl_y = self.size[1] - (y + h)
        data = self.fbo.read(viewport=(x, gl_y, w, h), components=4, dtype="f1")

        surface = pygame.image.frombuffer(data, (w, h), "RGBA")
        return pygame.transform.flip(surface, False, True)
