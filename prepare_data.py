from pathlib import Path
import random
import shutil

random.seed(42)

# Попробуем найти реальную папку, где лежат классы
candidates = [
    Path("data/PlantVillage"),
    Path("data/PlantVillage/PlantVillage"),
    Path("data/PlantVillage/PlantVillage Dataset"),
    Path("data/PlantVillage/PlantVillage-Dataset"),
]

SRC = None
for c in candidates:
    if c.exists() and any(p.is_dir() for p in c.iterdir()):
        SRC = c
        break

if SRC is None:
    raise FileNotFoundError("Не нашёл папку датасета. Проверь путь data/PlantVillage")

OUT = Path("data")

mapping = {
    "Tomato_healthy": "Healthy",
    "Tomato_Septoria_leaf_spot": "Leaf_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellowing",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Pest_damage",
}

print(f"✅ Датасет: {SRC.resolve()}")

# очистка старых папок (чтобы не смешивались)
shutil.rmtree(OUT / "train", ignore_errors=True)
shutil.rmtree(OUT / "val", ignore_errors=True)

for src_folder, target_class in mapping.items():
    src_path = SRC / src_folder

    if not src_path.exists():
        print(f"❌ Не найдено: {src_folder}")
        continue

    files = [f for f in src_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(files)

    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        out_dir = OUT / split_name / target_class
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, out_dir / f.name)

    print(f"✅ {target_class}: всего {len(files)} | train {len(train_files)} | val {len(val_files)}")

print("✅ Готово: data/train и data/val собраны")