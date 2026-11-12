"""Экспортирует Live2D-модель в наш собственный риг (rig.json.gz).

Официальное ядро (Cubism Core) в браузере читает .moc3 и отдаёт:
  * вершины каждого меша в нейтрали,
  * UV/индексы/порядок отрисовки,
  * прозрачность меша **в нейтральной позе** (важно: с учётом pose3.json!),
  * сэмплы деформации по каждому параметру (atMin/atMax).

SDK в приложении не нужен — Chrome используется только здесь, офлайн.

Запуск:
    python tools/l2d_export_rig.py /tmp/l2d/Mao --out assets/live2d/mao
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

PAGE = """<!doctype html><html><body><pre id="s">x</pre>
<script src="https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi.js@7.4.2/dist/pixi.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js"></script>
<script>
const R=(v,k=4)=>Array.from(v).map(x=>Math.round(x*10**k)/10**k);
PIXI.live2d.Live2DModel.from('/MODEL').then(async model=>{
 try{model.internalModel.physics=null;model.internalModel.eyeBlink=null;model.internalModel.breath=null;}catch(e){}
 // pose НЕ трогаем: именно он прячет вторую пару рук и второй посох
 model.internalModel.motionManager.stopAllMotions();
 for(let i=0;i<30;i++) model.update(1/60);
 const core=model.internalModel.coreModel, n=core.getDrawableCount();
 const pp=core._model.drawables.parentPartIndices;
 const draws=[];
 for(let i=0;i<n;i++) draws.push({
   part: pp ? pp[i] : -1,
   partOpacity: (pp && core._model.parts.opacities) ? core._model.parts.opacities[pp[i]] : 1,
   tex: core.getDrawableTextureIndices(i),
   order: (core.getDrawableRenderOrders(i)[i]),
   opacity: core.getDrawableOpacity(i),
   uv: R(core.getDrawableVertexUvs(i)),
   idx: Array.from(core.getDrawableVertexIndices(i)),
 });
 const ids=core._parameterIds.slice();
 const neutral=ids.map((_,i)=>core.getParameterValueByIndex(i));
 const reset=()=>ids.forEach((id,i)=>core.setParameterValueById(id,neutral[i]));
 const read=()=>{const o=[];for(let i=0;i<n;i++)o.push(...R(core.getDrawableVertexPositions(i)));return o;};
 reset(); core.update(); const base=read();
 const params=[];
 for(let i=0;i<ids.length;i++){
   const id=ids[i], mn=core._parameterMinimumValues[i], mx=core._parameterMaximumValues[i];
   reset(); core.setParameterValueById(id,mn); core.update(); const a=read();
   reset(); core.setParameterValueById(id,mx); core.update(); const b=read();
   let d=0; for(let k=0;k<base.length;k++) d=Math.max(d,Math.abs(a[k]-base[k]),Math.abs(b[k]-base[k]));
   if(d>1e-4) params.push({id,min:mn,max:mx,atMin:a,atMax:b,delta:Math.round(d*1000)/1000});
 }
 await fetch('/save',{method:'POST',body:JSON.stringify({draws,base,params})});
 document.getElementById('s').textContent='SAVED';
}).catch(e=>{document.getElementById('s').textContent='ERR:'+e;});
</script></body></html>
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Handler(SimpleHTTPRequestHandler):
    result: bytes | None = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        Handler.result = self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


ROOT = Path(".")


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="папка с моделью (должна содержать *.model3.json и pose3.json)")
    ap.add_argument("--out", required=True, help="куда положить rig.json.gz")
    args = ap.parse_args()

    src = Path(args.model).expanduser().resolve()
    model_json = sorted(src.rglob("*.model3.json"))[0]
    ROOT = model_json.parent
    rel = model_json.relative_to(ROOT).as_posix()

    page = PAGE.replace("/MODEL", "/" + rel)
    (ROOT / "_export_rig.html").write_text(page, encoding="utf-8")

    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
        "--virtual-time-budget=60000",
        f"http://127.0.0.1:{port}/_export_rig.html",
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)

    data = Handler.result
    server.shutdown()
    if not data:
        raise SystemExit("Chrome не вернул риг — смотри _export_rig.html и консоль")

    rig = json.loads(data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "rig.json.gz"
    with gzip.open(dst, "wb", compresslevel=6) as f:
        f.write(data)

    vis = sum(1 for d in rig["draws"] if d["opacity"] > 0.01)
    print(f"мешей: {len(rig['draws'])}, видимых: {vis}, параметров: {len(rig['params'])}")
    print(f"-> {dst} ({dst.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
