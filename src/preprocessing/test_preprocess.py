import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from preprocess_image import preprocess_image


DATASET_NAME = "Project-AgML/DIMPSAR_medicinal_plant_classification"


print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)
train = dataset["train"]

class_names = train.features["label"].names

selected_classes = [
    "Aloevera",
    "Amla",
    "Ashoka",
    "Ashwagandha",
    "Hibiscus",
    "Tulasi",
]


def get_first_image(class_name):

    class_id = class_names.index(class_name)

    for example in train:

        if example["label"] == class_id:
            return np.array(example["image"])

    return None


fig, axes = plt.subplots(
    len(selected_classes),
    3,
    figsize=(12, 4 * len(selected_classes))
)


for row, class_name in enumerate(selected_classes):

    print(f"Processing {class_name}...")

    image = get_first_image(class_name)

    original, segmented, mask, grayscale = preprocess_image(image)

    # Original
    axes[row, 0].imshow(original)
    axes[row, 0].set_title(
        f"{class_name} - Resized"
    )
    axes[row, 0].axis("off")

    # Mask
    axes[row, 1].imshow(
        mask,
        cmap="gray"
    )
    axes[row, 1].set_title(
        "Segmentation Mask"
    )
    axes[row, 1].axis("off")

    # Segmented
    axes[row, 2].imshow(segmented)
    axes[row, 2].set_title(
        "Segmented Leaf"
    )
    axes[row, 2].axis("off")


plt.suptitle(
    "LeafLens - Preprocessing Pipeline",
    fontsize=18
)

plt.tight_layout()

output_path = (
    "data/processed/"
    "preprocessing_test.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nPreprocessing test completed.")
print(f"Saved to: {output_path}")
