"""GL-версия приложения: окно и рендер в одном контексте, без чтения кадра.

Персонаж рисуется прямо в окно, UI — поверх текстурой. Никаких FBO-readback.
"""

import sys

import moderngl
import numpy as np
import pygame

import expressions
from main import AvatarBrain, BACKGROUND_COLOR, QWEN_CACHE_PATH
from nori_engine.live2d import Live2DRig

WIDTH, HEIGHT = 1000, 700
FPS = 60

UI_VERT = """
#version 330
in vec2 in_pos; in vec2 in_uv; out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

UI_FRAG = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_tex;
void main() { f_color = texture(u_tex, v_uv); }
"""

QUAD = np.array([
    -1, -1, 0, 1,  1, -1, 1, 1,  1, 1, 1, 0,
    -1, -1, 0, 1,  1, 1, 1, 0, -1, 1, 0, 0,
], dtype="f4")


def main():
    pygame.init()
    pygame.key.set_repeat(350, 35)
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Nori — GL")
    ctx = moderngl.create_context()
    print("GL:", ctx.info.get("GL_RENDERER"))

    rig = Live2DRig(
        "assets/live2d/mao/rig.json.gz",
        "assets/live2d/mao/texture_orig.png",
        ctx=ctx,
    )
    print("риг:", rig.vertex_count, "вершин")

    ui = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    ui_tex = ctx.texture((WIDTH, HEIGHT), 4)
    ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    prog = ctx.program(vertex_shader=UI_VERT, fragment_shader=UI_FRAG)
    vbo = ctx.buffer(QUAD.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "in_pos", "in_uv")])
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    font = pygame.font.SysFont("dejavusans", 20)
    small = pygame.font.SysFont("dejavusans", 14)
    brain = AvatarBrain(QWEN_CACHE_PATH)

    pose = dict(expressions.NEUTRAL)
    clock = pygame.time.Clock()
    text = ""
    cursor = 0
    status = "Введите реплику. Esc — выход."
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.TEXTINPUT:
                text = text[:cursor] + event.text + text[cursor:]
                cursor += len(event.text)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKSPACE and cursor > 0:
                    text = text[:cursor - 1] + text[cursor:]
                    cursor -= 1
                elif event.key == pygame.K_LEFT:
                    cursor = max(0, cursor - 1)
                elif event.key == pygame.K_RIGHT:
                    cursor = min(len(text), cursor + 1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and text.strip():
                    phrase = text.strip()
                    text, cursor = "", 0
                    result = brain.ask(phrase)
                    if result is None:
                        status = f"«{phrase}» — думает…"
                    else:
                        pose = result["params"] or pose
                        status = f"«{phrase}» → {result['desc']}"
        pygame.key.start_text_input()

        result = brain.poll()
        if result is not None:
            pose = result["params"] or pose
            status = f"«{result['text']}» → {result['desc']} [{result['source']}]"

        rig.set_params({
            "ParamMouthOpenY": pose["mouth_open"],
            "ParamMouthForm": pose.get("mouth_smile", 0.0),
            "ParamBrowLY": pose["left_brow_y"],
            "ParamBrowRY": pose["right_brow_y"],
            "ParamEyeLOpen": pose["left_eye_open"],
            "ParamEyeROpen": pose["right_eye_open"],
            "ParamAngleX": pose.get("head_x", 0.0) * 30.0,
            "ParamAngleY": -pose.get("head_y", 0.0) * 30.0,
            "ParamAngleZ": -pose.get("face_angle", 0.0) * 30.0,
        })

        bg = tuple(c / 255 for c in BACKGROUND_COLOR)
        ctx.screen.clear(*bg, 1.0)
        rig.draw_into(ctx.screen, WIDTH, HEIGHT, clear_color=None)

        ui.fill((0, 0, 0, 0))
        panel = pygame.Surface((WIDTH, 90), pygame.SRCALPHA)
        panel.fill((16, 22, 32, 220))
        ui.blit(panel, (0, HEIGHT - 90))
        ui.blit(font.render(status, True, (220, 230, 245)), (20, HEIGHT - 78))
        box = pygame.Rect(20, HEIGHT - 46, WIDTH - 40, 32)
        pygame.draw.rect(ui, (246, 248, 252), box, border_radius=6)
        pygame.draw.rect(ui, (70, 150, 240), box, 2, border_radius=6)
        ui.blit(font.render(text, True, (20, 28, 38)), (box.x + 10, box.y + 4))

        ui_tex.write(pygame.image.tostring(ui, "RGBA", False))
        ui_tex.use(0)
        prog["u_tex"].value = 0
        vao.render()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
