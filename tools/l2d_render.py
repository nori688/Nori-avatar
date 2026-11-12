"""Рендер Live2D-модели в PNG через официальное ядро и headless Chrome.

SDK в приложении не нужен: Chrome используется только здесь, как офлайн-экспортёр.
Скрипт сам находит .model3.json, проверяет версию moc3, поднимает локальный
HTTP-сервер, снимает нужные состояния и складывает их в папку.

Запуск:
    python tools/l2d_render.py <папка_или_zip> [--out DIR] [--size 500x700]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:transparent;overflow:hidden}canvas{display:block}</style>
</head><body>
<script src="https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi.js@7.4.2/dist/pixi.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js"></script>
<script>
const q = new URLSearchParams(location.search);
const modelPath = q.get('model');
const W = 500, H = 700;
const app = new PIXI.Application({ width: W, height: H, backgroundAlpha: 0, antialias: true });
document.body.appendChild(app.view);
PIXI.live2d.Live2DModel.from(modelPath).then(model => {
  app.stage.addChild(model);
  const b = model.getBounds();
  const s = Math.min(W / b.width, H / b.height) * 0.96;
  model.scale.set(s);
  model.x = -b.x * s + (W - b.width * s) / 2;
  model.y = -b.y * s + (H - b.height * s) / 2;
  const core = model.internalModel.coreModel;
  for (let i = 0; i < 20; i++) model.update(1/60);
  for (const [k, v] of q.entries()) {
    if (k.startsWith('p_')) core.setParameterValueById(k.slice(2), parseFloat(v));
  }
  // именно model.update, а не core.update: он пересобирает меш из новых параметров
  model.update(1/60);
  app.render();
  document.title = 'READY';
}).catch(e => { document.title = 'ERR:' + e; });
</script></body></html>
"""

DEFAULT_STATES = {
    "neutral": {"ParamMouthOpenY": 0.0},
    "smile": {"ParamMouthOpenY": 0.55, "ParamMouthForm": 0.8, "ParamEyeLOpen": 0.7, "ParamEyeROpen": 0.7},
    "angry": {"ParamBrowLY": -0.6, "ParamBrowRY": -0.6, "ParamEyeLOpen": 0.45, "ParamEyeROpen": 0.45},
    "turn": {"ParamAngleX": 30, "ParamAngleY": 12, "ParamAngleZ": 8},
}


def find_model(root: Path) -> Path:
    matches = sorted(root.rglob("*.model3.json"))
    if not matches:
        raise SystemExit(f"Не найден *.model3.json в {root}")
    return matches[0]


def moc_version(json_path: Path):
    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    moc = json_path.parent / cfg["FileReferences"]["Moc"]
    if not moc.exists():
        return None, moc
    header = moc.read_bytes()[:12]
    if header[:4] != b"MOC3":
        return None, moc
    return struct.unpack("<I", header[4:8])[0], moc


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(directory: Path, port: int):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="папка с моделью или zip")
    ap.add_argument("--out", default="_preview")
    ap.add_argument("--size", default="500x700")
    args = ap.parse_args()

    if not CHROME:
        raise SystemExit("Не найден google-chrome / chromium")

    src = Path(args.model).expanduser()
    work = src
    tmp = None
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="l2d_"))
        shutil.unpack_archive(str(src), str(tmp))
        work = tmp

    model_json = find_model(work)
    version, moc = moc_version(model_json)
    print(f"модель: {model_json.relative_to(work) if work in model_json.parents else model_json}")
    print(f"moc3: {moc.name}, версия формата: {version}")

    root = model_json.parent
    (root / "_viewer.html").write_text(PAGE, encoding="utf-8")
    port = free_port()
    serve(root, port)
    time.sleep(0.3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rel = model_json.relative_to(root).as_posix()
    w, h = args.size.split("x")

    for name, params in DEFAULT_STATES.items():
        query = "&".join(f"p_{k}={v}" for k, v in params.items())
        target = out / f"{moc.stem}_{name}.png"
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--default-background-color=00000000", f"--window-size={w},{h}",
            "--virtual-time-budget=15000", f"--screenshot={target}",
            f"http://127.0.0.1:{port}/_viewer.html?model=/{rel}&{query}",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        size = target.stat().st_size if target.exists() else 0
        print(f"  {name}: {size} байт -> {target}")

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
