import os
import time
import numpy as np
import cv2

from datasets import load_dataset
from skimage import feature
from skimage.filters import gabor

from src.preprocessing.preprocess_image import preprocess_image


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = (
    "Project-AgML/DIMPSAR_medicinal_plant_classification"
)

# False = test mode
# True  = process the complete dataset
FULL_DATASET_MODE = True

TEST_IMAGE_COUNT = 200

CHECKPOINT_INTERVAL = 100

OUTPUT_DIR = "data/processed"

if FULL_DATASET_MODE:
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "features_full.npz"
    )
else:
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "features_test_200.npz"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_lbp_features(image):
    lbp = feature.local_binary_pattern(
        image,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    lbp_hist = lbp_hist.astype(np.float32)

    lbp_hist /= (
        lbp_hist.sum() + 1e-8
    )

    return lbp_hist


def extract_glcm_features(image):
    quantized = (
        image / 256 * 32
    ).astype(np.uint8)

    quantized = np.clip(
        quantized,
        0,
        31
    )

    glcm = feature.graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )

    properties = [
        "dissimilarity",
        "contrast",
        "homogeneity",
        "energy",
        "correlation"
    ]

    values = []

    for prop in properties:
        value = feature.graycoprops(
            glcm,
            prop=prop
        )

        values.append(
            float(value[0, 0])
        )

    return np.array(
        values,
        dtype=np.float32
    )


def extract_gabor_features(image):
    image_float = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_AREA
    ).astype(
        np.float32
    ) / 255.0

    theta_values = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    frequency_values = [
        0.1,
        0.5
    ]

    features = []

    for theta in theta_values:
        for frequency in frequency_values:

            real, _ = gabor(
                image_float,
                frequency=frequency,
                theta=theta
            )

            features.append(
                float(np.mean(real))
            )

    return np.array(
        features,
        dtype=np.float32
    )


def extract_color_moments(image):
    channels = cv2.split(image)

    moments = []

    for channel in channels:

        channel = channel.astype(
            np.float32
        )

        mean = np.mean(channel)

        variance = np.var(channel)

        skewness = (
            np.mean(
                (channel - mean) ** 3
            )
            / (
                variance ** 1.5
                + 1e-8
            )
        )

        moments.extend([
            float(mean),
            float(variance),
            float(skewness)
        ])

    return np.array(
        moments,
        dtype=np.float32
    )


def extract_gradient_features(image):
    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        gradient_x ** 2
        + gradient_y ** 2
    )

    gradient_magnitude = cv2.normalize(
        gradient_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    gradient_magnitude = (
        gradient_magnitude.astype(
            np.uint8
        )
    )

    gradient_lbp = (
        extract_lbp_features(
            gradient_magnitude
        )
    )

    gradient_glcm = (
        extract_glcm_features(
            gradient_magnitude
        )
    )

    gradient_gabor = (
        extract_gabor_features(
            gradient_magnitude
        )
    )

    return np.concatenate([
        gradient_lbp,
        gradient_glcm,
        gradient_gabor
    ])


def extract_feature_vector(image):

    resized, segmented, mask, grayscale = (
        preprocess_image(image)
    )

    lbp_features = (
        extract_lbp_features(
            grayscale
        )
    )

    glcm_features = (
        extract_glcm_features(
            grayscale
        )
    )

    gabor_features = (
        extract_gabor_features(
            grayscale
        )
    )

    color_features = (
        extract_color_moments(
            segmented
        )
    )

    gradient_features = (
        extract_gradient_features(
            grayscale
        )
    )

    feature_vector = np.concatenate([
        lbp_features,
        glcm_features,
        gabor_features,
        gradient_features,
        color_features
    ])

    return feature_vector.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

def load_leaflens_dataset():

    print("=" * 60)
    print("LOADING LEAFLENS DATASET")
    print("=" * 60)

    dataset = load_dataset(
        DATASET_NAME
    )

    train = dataset["train"]

    class_names = (
        train.features["label"].names
    )

    print(
        f"Total dataset images: "
        f"{len(train)}"
    )

    print(
        f"Total classes: "
        f"{len(class_names)}"
    )

    return train, class_names


def select_test_images(
    dataset,
    target_count
):

    target_count = min(
        target_count,
        len(dataset)
    )

    selected = []

    for index in range(
        target_count
    ):
        selected.append(
            dataset[index]
        )

    return selected


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    all_features,
    all_labels,
    class_names,
    index
):

    checkpoint_file = os.path.join(
        OUTPUT_DIR,
        f"features_checkpoint_{index}.npz"
    )

    np.savez_compressed(
        checkpoint_file,
        X=np.array(
            all_features,
            dtype=np.float32
        ),
        y=np.array(
            all_labels,
            dtype=np.int64
        ),
        class_names=np.array(
            class_names
        )
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    dataset, class_names = (
        load_leaflens_dataset()
    )

    # --------------------------------------------------------
    # Select images
    # --------------------------------------------------------

    if FULL_DATASET_MODE:

        examples = dataset

        print(
            "\nFULL DATASET MODE"
        )

        print(
            f"Images to process: "
            f"{len(examples)}"
        )

    else:

        print(
            "\nTEST MODE"
        )

        examples = select_test_images(
            dataset,
            TEST_IMAGE_COUNT
        )

        print(
            f"Selected images: "
            f"{len(examples)}"
        )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print(
        "\nStarting feature extraction..."
    )

    all_features = []

    all_labels = []

    start_time = time.time()

    successful = 0

    failed = 0

    for index, example in enumerate(
        examples,
        start=1
    ):

        try:

            image = np.array(
                example["image"]
            )

            label = example["label"]

            features = (
                extract_feature_vector(
                    image
                )
            )

            if not np.all(
                np.isfinite(features)
            ):

                raise ValueError(
                    "Feature vector contains "
                    "NaN or infinite values."
                )

            all_features.append(
                features
            )

            all_labels.append(
                label
            )

            successful += 1

            print(
                f"[{index}/{len(examples)}] "
                f"{class_names[label]:20s} "
                f"Features: {len(features)}"
            )

        except Exception as error:

            failed += 1

            print(
                f"[{index}/{len(examples)}] "
                f"FAILED: {error}"
            )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        if (
            index % CHECKPOINT_INTERVAL == 0
        ):

            save_checkpoint(
                all_features,
                all_labels,
                class_names,
                index
            )

    # --------------------------------------------------------
    # Final arrays
    # --------------------------------------------------------

    X = np.array(
        all_features,
        dtype=np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int64
    )

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        class_names=np.array(
            class_names
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    if FULL_DATASET_MODE:

        print(
            "FULL FEATURE EXTRACTION COMPLETE"
        )

    else:

        print(
            "FEATURE EXTRACTION TEST COMPLETE"
        )

    print(
        "=" * 60
    )

    print(
        f"Successful images : "
        f"{successful}"
    )

    print(
        f"Failed images     : "
        f"{failed}"
    )

    print(
        f"Feature matrix    : "
        f"{X.shape}"
    )

    print(
        f"Feature vector size: "
        f"{X.shape[1]}"
    )

    print(
        f"Processing time   : "
        f"{elapsed:.2f} seconds"
    )

    if successful > 0:

        print(
            f"Average/image     : "
            f"{elapsed / successful:.3f} seconds"
        )

    print(
        "\nSaved features to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
