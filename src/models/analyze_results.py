import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CONFUSION_MATRIX_PATH = (
    BASE_DIR / "data" / "processed" / "models" / "confusion_matrix.npy"
)


# --------------------------------------------------
# Class names
# --------------------------------------------------

CLASS_NAMES = [
    "Aloevera",
    "Amla",
    "Amruta_Balli",
    "Arali",
    "Ashoka",
    "Ashwagandha",
    "Avacado",
    "Bamboo",
    "Basale",
    "Betel",
    "Betel_Nut",
    "Brahmi",
    "Castor",
    "Curry_Leaf",
    "Doddapatre",
    "Ekka",
    "Ganike",
    "Gauva",
    "Geranium",
    "Henna",
    "Hibiscus",
    "Honge",
    "Insulin",
    "Jasmine",
    "Lemon",
    "Lemon_grass",
    "Mango",
    "Mint",
    "Nagadali",
    "Neem",
    "Nithyapushpa",
    "Nooni",
    "Pappaya",
    "Pepper",
    "Pomegranate",
    "Raktachandini",
    "Rose",
    "Sapota",
    "Tulasi",
    "Wood_sorel",
]


# --------------------------------------------------
# Load confusion matrix
# --------------------------------------------------

if not CONFUSION_MATRIX_PATH.exists():
    raise FileNotFoundError(
        f"Confusion matrix not found:\n{CONFUSION_MATRIX_PATH}"
    )

cm = np.load(CONFUSION_MATRIX_PATH)

print("=" * 60)
print("LEAFLENS ERROR ANALYSIS")
print("=" * 60)

print(f"\nConfusion matrix shape: {cm.shape}")

if cm.shape != (40, 40):
    raise ValueError(
        f"Expected a 40x40 confusion matrix, got {cm.shape}"
    )


# --------------------------------------------------
# 1. Plot confusion matrix
# --------------------------------------------------

plt.figure(figsize=(16, 14))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("LeafLens - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    fontsize=7
)

plt.colorbar(label="Number of Images")

plt.tight_layout()

output_dir = BASE_DIR / "data" / "processed" / "analysis"
output_dir.mkdir(parents=True, exist_ok=True)

cm_path = output_dir / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix saved to:")
print(cm_path)


# --------------------------------------------------
# 2. Find major confusion pairs
# --------------------------------------------------

confusions = []

for true_idx in range(len(CLASS_NAMES)):

    for pred_idx in range(len(CLASS_NAMES)):

        # Ignore correct predictions
        if true_idx == pred_idx:
            continue

        count = cm[true_idx, pred_idx]

        if count > 0:

            confusions.append(
                (
                    int(count),
                    CLASS_NAMES[true_idx],
                    CLASS_NAMES[pred_idx]
                )
            )


# Sort from highest number of mistakes to lowest

confusions.sort(reverse=True)


# --------------------------------------------------
# 3. Print top confusion pairs
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP CONFUSION PAIRS")
print("=" * 60)

top_n = min(20, len(confusions))

for rank, (count, true_class, predicted_class) in enumerate(
    confusions[:top_n],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{true_class:20s} -> "
        f"{predicted_class:20s} : "
        f"{count} images"
    )


# --------------------------------------------------
# 4. Per-class recall
# --------------------------------------------------

print("\n" + "=" * 60)
print("LOWEST RECALL CLASSES")
print("=" * 60)

class_recall = []

for i, class_name in enumerate(CLASS_NAMES):

    total_actual = cm[i].sum()

    if total_actual == 0:
        recall = 0.0
    else:
        recall = cm[i, i] / total_actual

    class_recall.append(
        (
            recall,
            class_name,
            int(total_actual),
            int(cm[i, i])
        )
    )


class_recall.sort()


for rank, (recall, class_name, total, correct) in enumerate(
    class_recall[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{class_name:20s} "
        f"Recall: {recall:.2%} "
        f"({correct}/{total})"
    )


# --------------------------------------------------
# 5. Best recall classes
# --------------------------------------------------

print("\n" + "=" * 60)
print("HIGHEST RECALL CLASSES")
print("=" * 60)

for rank, (recall, class_name, total, correct) in enumerate(
    reversed(class_recall[-10:]),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{class_name:20s} "
        f"Recall: {recall:.2%} "
        f"({correct}/{total})"
    )


# --------------------------------------------------
# 6. Summary
# --------------------------------------------------

total_samples = cm.sum()
correct_predictions = np.trace(cm)

accuracy = correct_predictions / total_samples

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total test images : {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Incorrect predictions: {total_samples - correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")

print("\nError analysis complete.")
