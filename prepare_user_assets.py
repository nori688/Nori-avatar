"""Извлечение деталей аватара из сгенерированного JPEG (устарело; используйте export_psd.py)."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "teal"
SRC_PATH = Path("/home/nori/Загрузки/Gemini_Generated_Image_6vrmug6vrmug6vrm.jpeg")


def extract(src, name, box, close_bottom=False):
    im = src.crop(box)
    a = np.array(im).astype(int)
    mask = (a.max(2) - a.min(2) > 27) | (a.mean(2) < 140)
    mask = ndimage.binary_closing(mask, iterations=1)
    if close_bottom:
        mask[-2, 12:-12] = True
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    for i in range(1, count + 1):
        if (labels == i).sum() < 55:
            mask[labels == i] = False
    rgba = im.convert("RGBA")
    rgba.putalpha(Image.fromarray((mask * 255).astype("uint8")))
    rgba.save(OUT / (name + ".png"))


def main():
    if not SRC_PATH.exists():
        print(f"Примечание: Исходный JPEG не найден ({SRC_PATH}).")
        print("Для актуальных ассетов используйте `python export_psd.py`.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    src = Image.open(SRC_PATH).convert("RGB").resize((1568, 1589))

    extract(src, "base", (40, 170, 715, 869), close_bottom=True)
    extract(src, "hair", (785, 175, 1280, 995))
    extract(src, "back", (1290, 185, 1545, 995))
    extract(src, "left_eye", (500, 1240, 794, 1450))
    extract(src, "right_eye", (875, 1270, 1170, 1465))
    extract(src, "left_brow", (90, 1220, 385, 1300))
    extract(src, "right_brow", (90, 1350, 385, 1430))
    extract(src, "mouth", (1265, 1315, 1495, 1395))
    print(f"Ассеты сохранены в: {OUT}")


if __name__ == "__main__":
    main()
