import os
import time

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = (
    "Project-AgML/"
    "DIMPSAR_medicinal_plant_classification"
)

IMAGE_SIZE = (224, 224)

NUM_CLASSES = 40

# Fine-tuned model
MODEL_PATH = (
    "data/processed/cnn/"
    "mobilenetv3small_finetuned_best.keras"
)

# IMPORTANT:
# This is the SAME split used for all previous experiments.
SPLIT_PATH = (
    "data/processed/cnn/"
    "cnn_dataset_split.npz"
)

OUTPUT_DIR = (
    "data/processed/cnn"
)


# ============================================================
# Load model
# ============================================================

def load_leaflens_model():

    print("\n" + "=" * 70)
    print("LOADING FINE-TUNED LEAFLENS CNN")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Fine-tuned model not found:\n"
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "\nModel loaded successfully:"
    )

    print(
        MODEL_PATH
    )

    return model


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_image(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV3Small is being used with
    # include_preprocessing=False.
    #
    # Scale:
    # 0 - 255 --> -1 to +1

    image = (
        image / 127.5
    ) - 1.0

    return image


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 70)
    print("LEAFLENS FINE-TUNED CNN TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME
    )

    full_dataset = dataset[
        "train"
    ]

    print(
        f"Total dataset images: "
        f"{len(full_dataset)}"
    )

    # --------------------------------------------------------
    # Load saved split
    # --------------------------------------------------------

    if not os.path.exists(
        SPLIT_PATH
    ):

        raise FileNotFoundError(
            f"Dataset split not found:\n"
            f"{SPLIT_PATH}"
        )

    split_data = np.load(
        SPLIT_PATH,
        allow_pickle=True
    )

    test_indices = split_data[
        "test_indices"
    ]

    class_names = split_data[
        "class_names"
    ].tolist()

    print(
        f"\nTest images: "
        f"{len(test_indices)}"
    )

    print(
        f"Number of classes: "
        f"{len(class_names)}"
    )

    if len(test_indices) != 595:

        print(
            "\nWarning: Expected 595 test images, "
            f"but found {len(test_indices)}."
        )

    # --------------------------------------------------------
    # Load fine-tuned model
    # --------------------------------------------------------

    model = load_leaflens_model()

    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    print("\nPreparing test images...")

    test_images = []

    true_labels = []

    start_time = time.time()

    for count, index in enumerate(
        test_indices,
        start=1
    ):

        example = full_dataset[
            int(index)
        ]

        image = preprocess_image(
            example["image"]
        )

        test_images.append(
            image
        )

        true_labels.append(
            example["label"]
        )

        if count % 100 == 0:

            print(
                f"Prepared "
                f"{count}/"
                f"{len(test_indices)} "
                f"test images"
            )

    test_images = np.asarray(
        test_images,
        dtype=np.float32
    )

    true_labels = np.asarray(
        true_labels,
        dtype=np.int32
    )

    preparation_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTest image array shape: "
        f"{test_images.shape}"
    )

    print(
        f"Test image range: "
        f"{test_images.min():.4f} "
        f"to "
        f"{test_images.max():.4f}"
    )

    print(
        f"Preparation time: "
        f"{preparation_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING TEST PREDICTIONS")
    print("=" * 70)

    prediction_start = time.time()

    probabilities = model.predict(
        test_images,
        batch_size=32,
        verbose=1
    )

    prediction_time = (
        time.time()
        - prediction_start
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro"
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(
            NUM_CLASSES
        ),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(
            NUM_CLASSES
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        "cnn_finetuned_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LeafLens Fine-Tuned CNN - Test Evaluation\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Model: MobileNetV3Small Fine-Tuned\n"
        )

        file.write(
            f"Test images: "
            f"{len(test_indices)}\n"
        )

        file.write(
            f"Number of classes: "
            f"{NUM_CLASSES}\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy:.4f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1:.4f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            "-" * 60
            + "\n"
        )

        file.write(
            report
        )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    cm_path = os.path.join(
        OUTPUT_DIR,
        "cnn_finetuned_confusion_matrix.npy"
    )

    np.save(
        cm_path,
        cm
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "cnn_finetuned_test_metrics.npz"
    )

    np.savez_compressed(
        metrics_path,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        prediction_time=prediction_time,
        num_test_images=len(
            test_indices
        )
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINE-TUNED CNN TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Accuracy : "
        f"{accuracy:.2%}"
    )

    print(
        f"Macro F1      : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1   : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"\nPrediction time: "
        f"{prediction_time:.2f} seconds"
    )

    print(
        f"Images tested: "
        f"{len(test_indices)}"
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    print("\nSaved classification report:")
    print(report_path)

    print("\nSaved confusion matrix:")
    print(cm_path)

    print("\nSaved metrics:")
    print(metrics_path)

    print("\n" + "=" * 70)
    print("FINE-TUNED CNN TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
