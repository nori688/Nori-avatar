"""Экспорт слоёв из py.psd в assets/nori с генерацией манифеста координат."""
import json
from pathlib import Path
from PIL import Image
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parent
PSD_PATH = ROOT / "py.psd"
OUT_DIR = ROOT / "assets" / "nori"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_layer(parent, *names):
    """Ищет слой по пути из имен."""
    curr = parent
    for name in names:
        found = None
        for item in curr:
            if item.name == name:
                found = item
                break
        if found is None:
            raise KeyError(f"Слой '{name}' не найден в {[x.name for x in curr]}")
        curr = found
    return curr


def main():
    if not PSD_PATH.exists():
        print(f"Файл {PSD_PATH} не найден.")
        return

    print("Загрузка py.psd...")
    psd = PSDImage.open(PSD_PATH)
    canvas_w, canvas_h = psd.size
    print(f"Размер холста: {canvas_w}x{canvas_h}")

    # Центр головы аватара как референсная точка
    # По оси X симметрия строго по 1250, по оси Y центр лица около 650
    HEAD_ORIGIN = (1250, 650)
    HEAD_CENTER = HEAD_ORIGIN
    BODY_CENTER = (1250, 1600)
    extracted = {}

    def save_part(key, layer_or_img, bbox):
        filename = f"{key}.png"
        filepath = OUT_DIR / filename
        if hasattr(layer_or_img, "composite"):
            img = layer_or_img.composite()
        else:
            img = layer_or_img
        img.save(filepath)

        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        w = bx2 - bx1
        h = by2 - by1

        extracted[key] = {
            "file": filename,
            "width": w,
            "height": h,
            "bbox": [bx1, by1, bx2, by2],
            "center": [cx, cy],
            # Относительное смещение от единого центра головы
            "rel_center": [cx - HEAD_ORIGIN[0], cy - HEAD_ORIGIN[1]],
        }
        print(f"Сохранён: {key} -> {filepath.name} ({w}x{h})")

    # 1. Задние волосы
    back_hair = find_layer(psd, "後ろ")
    save_part("back_hair", back_hair, back_hair.bbox)

    # 2. Тело и одежда (верхний бюст)
    body_group = find_layer(psd, "服 身体")
    save_part("body", body_group, body_group.bbox)
    save_part("torso", find_layer(psd, "服 身体", "体"), find_layer(psd, "服 身体", "体").bbox)
    save_part("right_arm_lower", find_layer(psd, "服 身体", "腕下 右"), find_layer(psd, "服 身体", "腕下 右").bbox)
    save_part("left_arm_lower", find_layer(psd, "服 身体", "腕下 左"), find_layer(psd, "服 身体", "腕下 左").bbox)
    save_part("right_arm_upper", find_layer(psd, "服 身体", "腕上 右"), find_layer(psd, "服 身体", "腕上 右").bbox)
    save_part("left_arm_upper", find_layer(psd, "服 身体", "腕上 左"), find_layer(psd, "服 身体", "腕上 左").bbox)

    # 3. Ноги и отдельные детали одежды сохраняем отдельно, чтобы не
    # превращать полного персонажа в один неподвижный спрайт.
    save_part("left_leg", find_layer(psd, "足", "脚", "脚 左"), find_layer(psd, "足", "脚", "脚 左").bbox)
    save_part("right_leg", find_layer(psd, "足", "脚", "脚 右"), find_layer(psd, "足", "脚", "脚 右").bbox)
    leg_shadow = find_layer(psd, "足", "脚 影")
    save_part("leg_shadow", leg_shadow, leg_shadow.bbox)

    # 4. Шея и база лица
    neck_group = find_layer(psd, "輪郭")
    face_base = find_layer(psd, "輪郭", "顔")
    save_part("face_base", face_base, face_base.bbox)

    # Композит шеи
    neck_layers = [find_layer(psd, "輪郭", name) for name in ("首 2", "首", "首影")]
    nx1 = min(l.bbox[0] for l in neck_layers)
    ny1 = min(l.bbox[1] for l in neck_layers)
    nx2 = max(l.bbox[2] for l in neck_layers)
    ny2 = max(l.bbox[3] for l in neck_layers)
    neck_img = Image.new("RGBA", (nx2 - nx1, ny2 - ny1), (0, 0, 0, 0))
    for l in neck_layers:
        comp = l.composite()
        neck_img.paste(comp, (l.bbox[0] - nx1, l.bbox[1] - ny1), comp)
    save_part("neck", neck_img, (nx1, ny1, nx2, ny2))

    # 4. Брови (внимание: японское '右' = character's right = viewer's LEFT)
    left_brow = find_layer(psd, "顔", "まゆ", "右 眉")
    right_brow = find_layer(psd, "顔", "まゆ", "左 眉")
    save_part("left_brow", left_brow, left_brow.bbox)
    save_part("right_brow", right_brow, right_brow.bbox)

    # 5. Белки глаз
    left_white = find_layer(psd, "顔", "白", "右 白")
    right_white = find_layer(psd, "顔", "白", "左 白")
    save_part("left_eye_white", left_white, left_white.bbox)
    save_part("right_eye_white", right_white, right_white.bbox)

    # 6. Зрачки
    left_iris = find_layer(psd, "顔", "目", "右 眼球")
    right_iris = find_layer(psd, "顔", "目", "左 眼球")
    save_part("left_eye_iris", left_iris, left_iris.bbox)
    save_part("right_eye_iris", right_iris, right_iris.bbox)

    # 7. Верхние ресницы / веки
    left_upper = find_layer(psd, "顔", "目 上", "右 上")
    right_upper = find_layer(psd, "顔", "目 上", "左 上")
    save_part("left_eyelid_upper", left_upper, left_upper.bbox)
    save_part("right_eyelid_upper", right_upper, right_upper.bbox)

    # 8. Нижние веки
    left_lower = find_layer(psd, "顔", "目 下", "右 上")
    right_lower = find_layer(psd, "顔", "目 下", "左 上")
    save_part("left_eyelid_lower", left_lower, left_lower.bbox)
    save_part("right_eyelid_lower", right_lower, right_lower.bbox)

    # 9. Нос
    nose = find_layer(psd, "顔", "鼻")
    save_part("nose", nose, nose.bbox)

    # 10. Рот (композит и части)
    mouth_group = find_layer(psd, "顔", "口")
    save_part("mouth", mouth_group, mouth_group.bbox)
    mouth_inside = find_layer(psd, "顔", "口", "口 中")
    mouth_upper = find_layer(psd, "顔", "口", "口 上")
    mouth_lower = find_layer(psd, "顔", "口", "口 下")
    save_part("mouth_inside", mouth_inside, mouth_inside.bbox)
    save_part("mouth_upper", mouth_upper, mouth_upper.bbox)
    save_part("mouth_lower", mouth_lower, mouth_lower.bbox)

    # 11. Передняя челка (композит и отдельные пряди)
    front_hair = find_layer(psd, "前")
    save_part("front_hair", front_hair, front_hair.bbox)

    hair_left = find_layer(psd, "前", "右", "髪 左")
    hair_right = find_layer(psd, "前", "左", "髪 右")
    hair_mid = find_layer(psd, "前", "中", "髪 中")
    save_part("front_hair_left", hair_left, hair_left.bbox)
    save_part("front_hair_right", hair_right, hair_right.bbox)
    save_part("front_hair_mid", hair_mid, hair_mid.bbox)

    manifest = {
        "canvas_size": list(psd.size),
        "head_center": list(HEAD_CENTER),
        "body_center": list(BODY_CENTER),
        "head_origin": list(HEAD_ORIGIN),
        "layers": extracted,
    }

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nМанифест успешно сохранён в {manifest_path}")


if __name__ == "__main__":
    main()
