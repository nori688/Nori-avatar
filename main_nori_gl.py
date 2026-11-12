"""GPU-окно Nori с единым OpenGL-контекстом для аватара и интерфейса."""

import os
import sys

import moderngl
import numpy as np
import pygame

import expressions
from main import (
    AvatarBrain,
    BACKGROUND_COLOR,
    FPS,
    NoriAvatar,
    QWEN_CACHE_PATH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    draw_text_interface,
)


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_tex;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""

QUAD = np.array([
    -1, -1, 0, 1,  1, -1, 1, 1,  1, 1, 1, 0,
    -1, -1, 0, 1,  1, 1, 1, 0, -1, 1, 0, 0,
], dtype="f4")


def main():
    pygame.init()
    pygame.key.set_repeat(350, 35)
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Nori — GPU Avatar")
    pygame.key.start_text_input()

    ctx = moderngl.create_context()
    print(f"[Nori] общий OpenGL-контекст: {ctx.info.get('GL_RENDERER')}")

    os.environ["NORI_SHARED_GPU"] = "1"
    nori = NoriAvatar((SCREEN_WIDTH, SCREEN_HEIGHT))
    print("[Nori] кадр компонуется и выводится через общий GPU-контекст")

    frame = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    texture = ctx.texture((SCREEN_WIDTH, SCREEN_HEIGHT), 4)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    vertex_buffer = ctx.buffer(QUAD.tobytes())
    vertex_array = ctx.vertex_array(program, [(vertex_buffer, "2f 2f", "in_pos", "in_uv")])

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("dejavusans", 20)
    small_font = pygame.font.SysFont("dejavusans", 14)
    brain = AvatarBrain(QWEN_CACHE_PATH)
    input_text = ""
    cursor_pos = 0
    last_output = None
    status = "Введите реплику — решение принимает модель."
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
                elif event.key == pygame.K_F2:
                    nori.reset()
                    last_output = None
                    status = "Нейтральная поза"
                elif event.key == pygame.K_h:
                    status = f"Чёлка: {nori.cycle_fringe()}"
                elif event.key == pygame.K_BACKSPACE and cursor_pos:
                    input_text = input_text[:cursor_pos - 1] + input_text[cursor_pos:]
                    cursor_pos -= 1
                elif event.key == pygame.K_LEFT:
                    cursor_pos = max(0, cursor_pos - 1)
                elif event.key == pygame.K_RIGHT:
                    cursor_pos = min(len(input_text), cursor_pos + 1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and input_text.strip():
                    phrase = input_text.strip()
                    input_text = ""
                    cursor_pos = 0
                    result = brain.ask(phrase)
                    if result is None:
                        status = f"«{phrase}» — Qwen обдумывает…"
                    else:
                        last_output = result["params"] or last_output or dict(expressions.NEUTRAL)
                        nori.apply_model_output(last_output, result["seqs"])
                        status = f"«{phrase}» → {result['desc']}"

        result = brain.poll()
        if result is not None:
            last_output = result["params"] or last_output or dict(expressions.NEUTRAL)
            nori.apply_model_output(last_output, result["seqs"])
            status = f"«{result['text']}» → {result['desc']} [{result['source']}]"

        nori.update(delta_time)
        nori.draw(frame)
        draw_text_interface(
            frame, font, small_font, input_text, cursor_pos, status, last_output,
            nori.NAME, brain.backend_label, brain.busy,
            NoriAvatar.FRINGE_MODES[nori.fringe_mode],
        )
        ctx.screen.clear(*(component / 255 for component in BACKGROUND_COLOR), 1.0)
        texture.write(pygame.image.tostring(frame, "RGBA", False))
        texture.use(0)
        program["u_tex"].value = 0
        vertex_array.render()
        pygame.display.flip()

    pygame.key.stop_text_input()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
