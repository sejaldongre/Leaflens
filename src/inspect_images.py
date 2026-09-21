from datasets import load_dataset
import matplotlib.pyplot as plt
import math

DATASET_NAME = "Project-AgML/DIMPSAR_medicinal_plant_classification"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train = dataset["train"]

class_names = train.features["label"].names

# Select 12 different plant classes
selected_classes = [
    "Aloevera",
    "Amla",
    "Ashoka",
    "Ashwagandha",
    "Bamboo",
    "Betel",
    "Castor",
    "Curry_Leaf",
    "Hibiscus",
    "Mango",
    "Neem",
    "Tulasi",
]

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for i, class_name in enumerate(selected_classes):
    class_id = class_names.index(class_name)

    # Find first image belonging to this class
    for example in train:
        if example["label"] == class_id:
            image = example["image"]
            break

    axes[i].imshow(image)
    axes[i].set_title(class_name, fontsize=12)
    axes[i].axis("off")

plt.suptitle(
    "LeafLens - Sample Images from Medicinal Plant Dataset",
    fontsize=18
)

plt.tight_layout()

output_path = "data/processed/sample_leaf_images.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nSample visualization saved to:")
print(output_path)
