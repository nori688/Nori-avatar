"""Быстрая самопроверка без окна: роутер, таймлайны, отрисовка.

Запуск: .venv-avatar/bin/python selftest.py
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pygame

import expressions
from semantic_router import SemanticRouter
from timeline import SEQUENCES, Timeline

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail}")
        FAILURES.append(label)


def test_router():
    print("Локальный роутер:")
    router = SemanticRouter()
    cases = [
        # (фраза, ожидаемая анимация | None, ожидаемый пресет | None)
        ("улыбнись", None, "smile"),
        ("посмотри налево", None, "look_left"),
        ("посмотри налево и улыбнись", None, "look_left"),
        ("кивни", "nod", None),
        ("закати глаза", "eye_roll", None),
        ("привет", "happy_nod", None),
        ("не злись", None, None),
        ("мне скучно", None, "bored"),
        ("подмигни правым глазом", None, "wink_right"),
        ("ты серьезно?", None, "skeptical"),
        ("покачай головой", "head_shake", None),
        ("смутись", "embarrassed", None),
        ("я в ярости", None, "angry"),
        ("кивни и подмигни", "nod", None),
    ]
    params, _desc, _score, _seqs = router.route("не злись")
    check(
        "«не злись» не включает злость",
        params is not None and params["left_brow_y"] < 0.2 and params["right_brow_y"] < 0.2,
        f"брови {params and round(params['left_brow_y'], 2)}",
    )

    for text, expected_seq, expected_preset in cases:
        params, desc, _score, seqs = router.route(text)
        if expected_seq:
            check(f"«{text}» → анимация {expected_seq}", expected_seq in (seqs or []), f"получено {seqs}")
        if expected_preset:
            difference = 0.0 if params is None else sum(
                abs(params[key] - expressions.NEUTRAL[key]) for key in expressions.NEUTRAL
            )
            ok = difference < 0.05 if expected_preset == "neutral" else difference > 0.25
            check(
                f"«{text}» → пресет {expected_preset}",
                ok,
                f"отклонение от нейтрали {difference:.2f}",
            )


def test_timelines():
    print("Таймлайны:")
    for name in ("nod", "slow_nod", "head_shake", "happy_nod", "sigh", "thinking"):
        timeline = Timeline(SEQUENCES[name])
        peak_head_y = 0.0
        peak_angle = 0.0
        for _ in range(400):
            frame = timeline.advance(1 / 60)
            if frame is None:
                break
            peak_head_y = max(peak_head_y, abs(frame.get("head_y", 0.0)))
            peak_angle = max(peak_angle, abs(frame.get("face_angle", 0.0)))
        check(
            f"{name}: голова реально двигается",
            peak_head_y > 0.3 or peak_angle > 0.3,
            f"head_y={peak_head_y:.2f} angle={peak_angle:.2f}",
        )

    # Моргание не должно «замыливаться» сглаживанием
    timeline = Timeline(SEQUENCES["double_blink"])
    minimum = 1.0
    for _ in range(120):
        frame = timeline.advance(1 / 60)
        if frame is None:
            break
        minimum = min(minimum, frame["left_eye_open"])
    check("double_blink полностью закрывает глаз", minimum < 0.05, f"min={minimum:.2f}")


def test_render():
    print("Отрисовка:")
    pygame.init()
    screen = pygame.display.set_mode((1000, 700))
    import main as app

    out = ROOT / "_preview"
    out.mkdir(exist_ok=True)

    nori = app.NoriAvatar(screen.get_size())
    avatars = [nori, app.ClassicAvatar(screen.get_size()), app.PixelSmiley(screen.get_size())]

    poses = {
        "neutral": expressions.NEUTRAL,
        "smile": expressions.blend_presets([("smile", 0.9)]),
        "angry": expressions.blend_presets([("angry", 0.9)]),
        "surprise": expressions.blend_presets([("surprise", 0.95)]),
        "wink": expressions.blend_presets([("wink_left", 1.0)]),
        "talk": expressions.blend_presets([("talk", 0.9)]),
    }

    for avatar in avatars:
        name = type(avatar).__name__
        for pose_name, params in poses.items():
            avatar.apply_model_output(params)
            avatar.parameters = dict(avatar.target_parameters)
            avatar.draw(screen)
            pygame.image.save(screen, str(out / f"{name}_{pose_name}.png"))

    # Открытый рот должен быть заметно выше закрытого
    nori.apply_model_output(expressions.blend_presets([("talk", 1.0)]))
    nori.parameters = dict(nori.target_parameters)
    nori.draw(screen)
    pygame.image.save(screen, str(out / "NoriAvatar_talk_max.png"))

    check("кадры сохранены", len(list(out.glob('*.png'))) >= 18, f"{len(list(out.glob('*.png')))} файлов")
    pygame.quit()


def main():
    test_router()
    test_timelines()
    test_render()
    print()
    if FAILURES:
        print(f"ПРОВАЛЕНО: {len(FAILURES)}")
        for item in FAILURES:
            print("  -", item)
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
