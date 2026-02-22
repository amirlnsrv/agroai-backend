import io
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# -------------------------
# Загрузка модели
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model.pth"  # backend/model.pth

ckpt = torch.load(MODEL_PATH, map_location="cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Восстанавливаем имена классов
idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}

# -------------------------
# Трансформация изображения
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Рекомендации
# -------------------------

RECOMMENDATIONS = {
    "Healthy": [
        "Лист выглядит здоровым — продолжайте обычный уход.",
        "Следите за поливом: без пересушивания и без переувлажнения.",
        "Периодически осматривайте растение для раннего выявления проблем."
    ],
    "Leaf_spot": [
        "Удалите поражённые листья и утилизируйте их.",
        "Снизьте влажность и улучшите вентиляцию.",
        "При необходимости обработайте фунгицидом по инструкции."
    ],
    "Yellowing": [
        "Проверьте режим полива и дренаж.",
        "Проверьте питание (возможно дефицит азота или железа).",
        "Осмотрите растение на вирусы и вредителей."
    ],
    "Pest_damage": [
        "Осмотрите нижнюю сторону листьев.",
        "Используйте мыльный раствор или инсектицид при необходимости.",
        "Изолируйте растение от других культур."
    ],
}

# Красивые названия для ответа
PRETTY_NAMES = {
    "Leaf_spot": "Leaf spot",
    "Pest_damage": "Pest damage",
}

# -------------------------
# Предсказание
# -------------------------

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    label_raw = idx_to_class[idx]
    label_pretty = PRETTY_NAMES.get(label_raw, label_raw)

    return {
        "label": label_pretty,
        "confidence": round(conf, 3),
        "recommendations": RECOMMENDATIONS.get(label_raw, [])
    }