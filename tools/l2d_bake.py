"""Запекает Live2D-модель в набор состояний для нашего движка.

Снимает полноростовые кадры при замороженной физике/дыхании, чтобы кадры
отличались только лицом — тогда в приложении их можно смешивать по альфе.

Запуск:
    python tools/l2d_bake.py <zip|папка> --name blonde --out assets/live2d/blonde
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

STATES = {
    "neutral":    {},
    "mouth_half": {"ParamMouthOpenY": 0.5},
    "mouth_open": {"ParamMouthOpenY": 1.0},
    "closed":     {"ParamEyeLOpen": 0.0, "ParamEyeROpen": 0.0},
    "smile":      {"ParamMouthForm": 1.0, "ParamMouthOpenY": 0.35,
                   "ParamEyeLSmile": 1.0, "ParamEyeRSmile": 1.0},
    "angry":      {"ParamBrowLY": -0.8, "ParamBrowRY": -0.8,
                   "ParamBrowLAngle": -1.0, "ParamBrowRAngle": -1.0,
                   "ParamMouthForm": -0.7, "ParamMouthOpenY": 0.15},
    "surprised":  {"ParamMouthOpenY": 0.85, "ParamBrowLY": 0.9, "ParamBrowRY": 0.9},
    "blush":      {"ParamCheek": 1.0},
}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:transparent;overflow:hidden}canvas{display:block}</style>
</head><body>
<script src="https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi.js@7.4.2/dist/pixi.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js"></script>
<script>
const q = new URLSearchParams(location.search);
const W = 900, H = 1400;
const app = new PIXI.Application({ width: W, height: H, backgroundAlpha: 0, antialias: true });
document.body.appendChild(app.view);
PIXI.live2d.Live2DModel.from(q.get('model')).then(model => {
  app.stage.addChild(model);
  // замораживаем всё, что живёт само по себе
  try {
    model.internalModel.physics = null;
    model.internalModel.eyeBlink = null;
    model.internalModel.breath = null;
    model.internalModel.pose = null;
  } catch (e) {}
  model.internalModel.motionManager.stopAllMotions();
  const b = model.getBounds();
  const s = Math.min(W / b.width, H / b.height) * 0.98;
  model.scale.set(s);
  model.x = -b.x * s + (W - b.width * s) / 2;
  model.y = -b.y * s + (H - b.height * s) / 2;
  const core = model.internalModel.coreModel;
  for (let i = 0; i < 30; i++) model.update(1/60);
  core.loadParameters();
  for (const [k, v] of q.entries()) {
    if (k.startsWith('p_')) core.setParameterValueById(k.slice(2), parseFloat(v));
  }
  core.saveParameters();
  model.update(1/60);
  app.render();
  document.title = 'READY';
}).catch(e => document.title = 'ERR:' + e);
</script></body></html>
"""


def find_model(root: Path) -> Path:
    matches = sorted(root.rglob("*.model3.json"))
    if not matches:
        raise SystemExit(f"Не найден *.model3.json в {root}")
    return matches[0]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", default="model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", default="900x1400")
    args = ap.parse_args()

    if not CHROME:
        raise SystemExit("Не найден google-chrome")

    src = Path(args.model).expanduser()
    tmp = None
    work = src
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="l2d_bake_"))
        shutil.unpack_archive(str(src), str(tmp))
        work = tmp

    model_json = find_model(work)
    root = model_json.parent
    out = Path(args.out) if args.out else Path("assets/live2d") / args.name
    out.mkdir(parents=True, exist_ok=True)

    (root / "_bake.html").write_text(PAGE, encoding="utf-8")
    port = free_port()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)
        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)

    w, h = args.size.split("x")
    rel = model_json.relative_to(root).as_posix()
    manifest = {"name": args.name, "size": [int(w), int(h)], "states": {}}

    for state, params in STATES.items():
        query = "&".join(f"p_{k}={v}" for k, v in params.items())
        target = out / f"{state}.png"
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--default-background-color=00000000", f"--window-size={w},{h}",
            "--virtual-time-budget=15000", f"--screenshot={target}",
            f"http://127.0.0.1:{port}/_bake.html?model=/{rel}&{query}",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        size = target.stat().st_size if target.exists() else 0
        manifest["states"][state] = {"params": params, "file": f"{state}.png", "bytes": size}
        print(f"  {state:12} {size:>8} байт")

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("готово:", out)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
