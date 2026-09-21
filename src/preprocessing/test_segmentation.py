import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET_NAME = "Project-AgML/DIMPSAR_medicinal_plant_classification"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train = dataset["train"]

class_names = train.features["label"].names

# Different plant types with different backgrounds/shapes
selected_classes = [
    "Aloevera",
    "Amla",
    "Ashoka",
    "Ashwagandha",
    "Betel",
    "Hibiscus",
]


def hsv_segmentation(image):
    """
    Segment the leaf using HSV saturation + color information.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Saturation helps separate vegetation from many dull backgrounds
    saturation = hsv[:, :, 1]

    # Green channel information
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    excess_green = 2 * g - r - b

    saturation_mask = saturation > 45
    green_mask = excess_green > 10

    # Combine the two signals
    mask = np.logical_or(saturation_mask, green_mask).astype(np.uint8) * 255

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask


def grabcut_segmentation(image):
    """
    Use GrabCut with a border rectangle as the initial estimate.
    """

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width = image_bgr.shape[:2]

    mask = np.zeros(
        image_bgr.shape[:2],
        np.uint8
    )

    # Start with the border as definite background
    border = 10

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Everything inside is initially considered probable foreground
    mask[border:-border, border:-border] = cv2.GC_PR_FGD

    # GrabCut models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_bgr,
            mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        result = cv2.bitwise_and(
            image,
            image,
            mask=final_mask
        )

        return result, final_mask

    except cv2.error:
        return image.copy(), np.ones(
            image.shape[:2],
            dtype=np.uint8
        ) * 255


def find_image(class_id):
    """
    Find the first image belonging to a class.
    """

    for example in train:
        if example["label"] == class_id:
            return np.array(example["image"])

    return None


# Create comparison figure
fig, axes = plt.subplots(
    len(selected_classes),
    3,
    figsize=(15, 5 * len(selected_classes))
)

for row, class_name in enumerate(selected_classes):

    print(f"Processing: {class_name}")

    class_id = class_names.index(class_name)

    image = find_image(class_id)

    hsv_result, hsv_mask = hsv_segmentation(image)
    grabcut_result, grabcut_mask = grabcut_segmentation(image)

    # Original
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{class_name}\nOriginal")
    axes[row, 0].axis("off")

    # HSV
    axes[row, 1].imshow(hsv_result)
    axes[row, 1].set_title("HSV + Color Segmentation")
    axes[row, 1].axis("off")

    # GrabCut
    axes[row, 2].imshow(grabcut_result)
    axes[row, 2].set_title("GrabCut Segmentation")
    axes[row, 2].axis("off")


plt.suptitle(
    "LeafLens - Segmentation Comparison",
    fontsize=18
)

plt.tight_layout()

output_path = "data/processed/segmentation_comparison.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 60)
print("SEGMENTATION TEST COMPLETE")
print("=" * 60)
print(f"Saved to: {output_path}")
