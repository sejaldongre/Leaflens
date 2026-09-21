from datasets import load_dataset
from collections import Counter

DATASET_NAME = "Project-AgML/DIMPSAR_medicinal_plant_classification"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train = dataset["train"]

# Class names
class_names = train.features["label"].names

print("\n" + "=" * 60)
print("LEAFLENS DATASET OVERVIEW")
print("=" * 60)

print(f"Total images : {len(train)}")
print(f"Total classes: {len(class_names)}")

# Class distribution
label_counts = Counter(train["label"])

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for label_id, count in sorted(label_counts.items()):
    print(f"{class_names[label_id]:20s} : {count}")

# Statistics
counts = list(label_counts.values())

print("\n" + "=" * 60)
print("CLASS BALANCE")
print("=" * 60)

print(f"Minimum images/class: {min(counts)}")
print(f"Maximum images/class: {max(counts)}")
print(f"Average images/class: {sum(counts) / len(counts):.2f}")

# Image dimensions
print("\n" + "=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

sample_images = train.select(range(min(100, len(train))))

widths = []
heights = []

for example in sample_images:
    image = example["image"]
    widths.append(image.width)
    heights.append(image.height)

print(f"Sampled images: {len(sample_images)}")
print(f"Width range   : {min(widths)} - {max(widths)} pixels")
print(f"Height range  : {min(heights)} - {max(heights)} pixels")
