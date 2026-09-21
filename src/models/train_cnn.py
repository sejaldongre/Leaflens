import os
import random
import time

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = (
    "Project-AgML/"
    "DIMPSAR_medicinal_plant_classification"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

RANDOM_STATE = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

NUM_CLASSES = 40

EPOCHS = 10

LEARNING_RATE = 1e-3

OUTPUT_DIR = "data/processed/cnn"


# ============================================================
# Reproducibility
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Build model
# ============================================================

def build_model():

    print("\n" + "=" * 70)
    print("BUILDING MOBILENETV3SMALL")
    print("=" * 70)

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False
    )

    # Freeze pretrained ImageNet layers
    base_model.trainable = False

    print("\nBase model:")
    print("MobileNetV3Small")

    print(
        f"Base model parameters: "
        f"{base_model.count_params():,}"
    )

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),
        name="leaf_image"
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="plant_classifier"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="LeafLens_MobileNetV3Small"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LEAFLENS CNN TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME
    )

    full_dataset = dataset["train"]

    print(
        f"Total images: "
        f"{len(full_dataset)}"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = np.array(
        full_dataset["label"]
    )

    class_names = (
        full_dataset
        .features["label"]
        .names
    )

    num_classes = len(
        class_names
    )

    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {num_classes}."
        )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    indices = np.arange(
        len(full_dataset)
    )

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1 - TRAIN_SIZE),
        stratify=labels,
        random_state=RANDOM_STATE
    )

    temp_labels = labels[
        temp_indices
    ]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_STATE
    )

    train_indices = np.sort(
        train_indices
    )

    val_indices = np.sort(
        val_indices
    )

    test_indices = np.sort(
        test_indices
    )

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Training   : "
        f"{len(train_indices)}"
    )

    print(
        f"Validation : "
        f"{len(val_indices)}"
    )

    print(
        f"Test       : "
        f"{len(test_indices)}"
    )

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    train_set = set(
        train_indices
    )

    val_set = set(
        val_indices
    )

    test_set = set(
        test_indices
    )

    if train_set & val_set:
        raise ValueError(
            "Training/validation overlap detected."
        )

    if train_set & test_set:
        raise ValueError(
            "Training/test overlap detected."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation/test overlap detected."
        )

    print(
        "\nNo data leakage detected."
    )

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # MobileNetV3Small is being used with:
        #
        # include_preprocessing=False
        #
        # Therefore we manually scale:
        #
        # 0 - 255  -->  -1 to +1
        # ----------------------------------------------------

        image = (
            image / 127.5
        ) - 1.0

        return image

    # --------------------------------------------------------
    # Verify preprocessing
    # --------------------------------------------------------

    print("\nTesting image preprocessing...")

    sample = full_dataset[
        int(train_indices[0])
    ]

    processed_sample = preprocess_image(
        sample["image"]
    )

    print(
        f"Processed image shape: "
        f"{processed_sample.shape}"
    )

    print(
        f"Processed image dtype: "
        f"{processed_sample.dtype}"
    )

    print(
        f"Processed image range: "
        f"{processed_sample.min():.4f} "
        f"to "
        f"{processed_sample.max():.4f}"
    )

    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    class LeafDataGenerator(
        tf.keras.utils.Sequence
    ):

        def __init__(
            self,
            dataset,
            indices,
            labels,
            batch_size,
            shuffle=False,
            augment=False
        ):

            self.dataset = dataset

            self.indices = np.array(
                indices
            )

            self.labels = labels

            self.batch_size = batch_size

            self.shuffle = shuffle

            self.augment = augment

            self.order = np.arange(
                len(self.indices)
            )

            self.on_epoch_end()

        def __len__(self):

            return int(
                np.ceil(
                    len(self.indices)
                    / self.batch_size
                )
            )

        def __getitem__(
            self,
            batch_index
        ):

            start = (
                batch_index
                * self.batch_size
            )

            end = min(
                start
                + self.batch_size,
                len(self.indices)
            )

            batch_positions = self.order[
                start:end
            ]

            batch_indices = self.indices[
                batch_positions
            ]

            images = []

            batch_labels = []

            for index in batch_indices:

                example = self.dataset[
                    int(index)
                ]

                image = preprocess_image(
                    example["image"]
                )

                images.append(
                    image
                )

                batch_labels.append(
                    self.labels[index]
                )

            images = np.asarray(
                images,
                dtype=np.float32
            )

            batch_labels = np.asarray(
                batch_labels,
                dtype=np.int32
            )

            return (
                images,
                batch_labels
            )

        def on_epoch_end(self):

            if self.shuffle:

                np.random.shuffle(
                    self.order
                )

    # --------------------------------------------------------
    # Create generators
    # --------------------------------------------------------

    print(
        "\nCreating data generators..."
    )

    train_generator = LeafDataGenerator(
        dataset=full_dataset,
        indices=train_indices,
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=False
    )

    val_generator = LeafDataGenerator(
        dataset=full_dataset,
        indices=val_indices,
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )

    print(
        f"Training batches   : "
        f"{len(train_generator)}"
    )

    print(
        f"Validation batches : "
        f"{len(val_generator)}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "mobilenetv3small_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        f"\nEpochs       : "
        f"{EPOCHS}"
    )

    print(
        f"Batch size   : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "\nBase MobileNetV3Small layers "
        "are frozen."
    )

    training_start = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    training_time = (
        time.time()
        - training_start
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Best validation result
    # --------------------------------------------------------

    best_epoch = int(
        np.argmax(
            history.history[
                "val_accuracy"
            ]
        )
    )

    best_val_accuracy = (
        history.history[
            "val_accuracy"
        ][best_epoch]
    )

    best_train_accuracy = (
        history.history[
            "accuracy"
        ][best_epoch]
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch + 1}"
    )

    print(
        f"Training accuracy at best epoch: "
        f"{best_train_accuracy:.2%}"
    )

    print(
        f"Validation accuracy at best epoch: "
        f"{best_val_accuracy:.2%}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        "mobilenetv3_training_history.npz"
    )

    np.savez_compressed(
        history_path,
        accuracy=np.array(
            history.history[
                "accuracy"
            ]
        ),
        val_accuracy=np.array(
            history.history[
                "val_accuracy"
            ]
        ),
        loss=np.array(
            history.history[
                "loss"
            ]
        ),
        val_loss=np.array(
            history.history[
                "val_loss"
            ]
        ),
        training_time=training_time
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = os.path.join(
        OUTPUT_DIR,
        "mobilenetv3small_frozen.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Save dataset split
    # --------------------------------------------------------

    split_path = os.path.join(
        OUTPUT_DIR,
        "cnn_dataset_split.npz"
    )

    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        class_names=np.array(
            class_names,
            dtype=object
        )
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nSaved best model:")
    print(
        best_model_path
    )

    print("\nSaved final model:")
    print(
        final_model_path
    )

    print("\nSaved training history:")
    print(
        history_path
    )

    print("\nSaved dataset split:")
    print(
        split_path
    )

    print("\n" + "=" * 70)
    print("FIRST CNN TRAINING PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
