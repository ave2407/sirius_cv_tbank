import os
import random
import shutil


src_root = "../../data/raw"

dst_root = "../../data/sample2"
os.makedirs(dst_root, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

all_imgs = []
for root, dirs, files in os.walk(src_root):
    for f in files:
        if os.path.splitext(f)[1].lower() in exts:
            all_imgs.append(os.path.join(root, f))

print("Найдено изображений:", len(all_imgs))

sampled = random.sample(all_imgs, min(500, len(all_imgs)))

for src in sampled:
    dst = os.path.join(dst_root, os.path.basename(src))
    base, ext = os.path.splitext(dst)
    i = 1
    while os.path.exists(dst):
        dst = f"{base}_{i}{ext}"
        i += 1
    shutil.copy2(src, dst)

print("Скопировано файлов:", len(sampled))
print("Папка:", dst_root)
