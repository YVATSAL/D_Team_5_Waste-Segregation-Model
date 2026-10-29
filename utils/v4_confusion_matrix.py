"""Recreate V4's validation split and export report-ready evaluation figures."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v4_confusion_matrix"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["Biodegradable", "Nonbiodegradable", "Recyclable"]
keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
paths, labels = [], []
for class_id, name in enumerate(CLASSES):
    valid = []
    for path in sorted((ROOT / "DATASET" / name).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        try:
            with Image.open(path) as im:
                im.load()
                im.convert("RGB")
            valid.append(str(path))
        except Exception:
            pass
    files = np.array(valid)
    rng.shuffle(files)
    held_out = files[int(len(files) * 0.8):]
    paths.extend(held_out.tolist())
    labels.extend([class_id] * len(held_out))
print(f"Validation images: {len(paths)}", flush=True)
model = keras.models.load_model(ROOT / "best_model_v4_fixed.keras", compile=False)
predictions = []
for start in range(0, len(paths), 32):
    batch = []
    for path in paths[start:start + 32]:
        with Image.open(path) as im:
            batch.append(np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32))
    predictions.extend(np.argmax(model(np.stack(batch), training=False).numpy(), axis=1).tolist())
    if start % 1024 == 0:
        print(f"Evaluated {min(start + 32, len(paths))}/{len(paths)}", flush=True)
cm = np.zeros((3, 3), dtype=np.int64)
np.add.at(cm, (labels, predictions), 1)
accuracy = float(np.mean(np.array(labels) == predictions))
np.savetxt(OUT / "confusion_matrix_counts.csv", cm, delimiter=",", fmt="%d", header=",".join(CLASSES), comments="")
report = 'Class                 Precision    Recall        F1   Support\n'
for i, name in enumerate(CLASSES):
    precision = cm[i, i] / max(cm[:, i].sum(), 1)
    recall = cm[i, i] / max(cm[i].sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    report += f'{name:22s} {precision:.4f}     {recall:.4f}    {f1:.4f}   {cm[i].sum()}\n'
(OUT / "classification_report.txt").write_text(report, encoding="utf-8")
(OUT / "evaluation_details.json").write_text(json.dumps({"model": "best_model_v4_fixed.keras", "split": "Per-class 80/20, seed 42; recreated from current DATASET using V4 notebook logic", "validation_images": len(labels), "accuracy": accuracy, "class_order": CLASSES, "confusion_matrix": cm.tolist()}, indent=2), encoding="utf-8")
with (OUT / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(["image", "actual", "predicted"])
    writer.writerows((str(Path(p).relative_to(ROOT)), CLASSES[y], CLASSES[pred]) for p, y, pred in zip(paths, labels, predictions))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
for normalized in [False, True]:
    values = cm / cm.sum(axis=1, keepdims=True) * 100 if normalized else cm
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(values, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage of actual class (%)" if normalized else "Number of images")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted class", ylabel="Actual class")
    ax.set_title("V4 — Validation Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{values[i,j]:.2f}%" if normalized else f"{cm[i,j]:,}", ha="center", va="center", fontsize=15, fontweight="bold", color="white" if values[i,j] > values.max()/2 else "#15283b")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", rotation_mode="anchor")
    fig.text(0.5, 0.025, f"Validation images: {len(labels):,}  |  Accuracy: {accuracy:.2%}\nPer-class 80/20 split · Random seed 42", ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    name = "v4_confusion_matrix_normalized" if normalized else "v4_confusion_matrix"
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)
print(cm, flush=True)
print(f"Accuracy: {accuracy:.6%}\n{report}\nSaved to {OUT}", flush=True)
