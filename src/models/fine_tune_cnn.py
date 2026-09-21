import os
import time

import numpy as np
import tensorflow as tf
from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = (
    "Project-AgML/"
    "DIMPSAR_medicinal_plant_classification"
)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_CLASSES = 40

EPOCHS = 10

# IMPORTANT:
# Fine-tuning uses a very small learning rate.
LEARNING_RATE = 1e-5

# Number of MobileNetV3Small layers to unfreeze
UNFREEZE_LAST_LAYERS = 30

MODEL_PATH = (
    "data/processed/cnn/"
    "mobilenetv3small_best.keras"
)

SPLIT_PATH = (
    "data/processed/cnn/"
    "cnn_dataset_split.npz"
)

OUTPUT_DIR = (
    "data/processed/cnn"
)

FINE_TUNED_MODEL_PATH = (
    "data/processed/cnn/"
    "mobilenetv3small_finetuned_best.keras"
)


# ============================================================
# Load dataset split
# ============================================================

def load_split():

    if not os.path.exists(SPLIT_PATH):

        raise FileNotFoundError(
            f"Dataset split not found:\n"
            f"{SPLIT_PATH}"
        )

    split_data = np.load(
        SPLIT_PATH,
        allow_pickle=True
    )

    train_indices = split_data[
        "train_indices"
    ]

    val_indices = split_data[
        "val_indices"
    ]

    test_indices = split_data[
        "test_indices"
    ]

    class_names = split_data[
        "class_names"
    ].tolist()

    return (
        train_indices,
        val_indices,
        test_indices,
        class_names
    )


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

    # MobileNetV3Small with
    # include_preprocessing=False
    #
    # Scale:
    # 0-255 --> -1 to +1

    image = (
        image / 127.5
    ) - 1.0

    return image


# ============================================================
# Data generator
# ============================================================

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
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.dataset = dataset

        self.indices = np.array(
            indices
        )

        self.labels = labels

        self.batch_size = batch_size

        self.shuffle = shuffle

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

        positions = self.order[
            start:end
        ]

        batch_indices = self.indices[
            positions
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


# ============================================================
# Configure fine-tuning
# ============================================================

def configure_fine_tuning(model):

    print("\n" + "=" * 70)
    print("CONFIGURING FINE-TUNING")
    print("=" * 70)

    # Find the MobileNetV3Small base model
    base_model = None

    for layer in model.layers:

        if (
            isinstance(
                layer,
                tf.keras.Model
            )
            and
            "MobileNetV3Small"
            in layer.name
        ):

            base_model = layer

            break

    if base_model is None:

        raise ValueError(
            "MobileNetV3Small base model "
            "could not be found."
        )

    # First freeze everything
    base_model.trainable = True

    total_layers = len(
        base_model.layers
    )

    print(
        f"\nTotal base-model layers: "
        f"{total_layers}"
    )

    # Freeze all layers
    for layer in base_model.layers:

        layer.trainable = False

    # Unfreeze only the last N layers
    layers_to_unfreeze = min(
        UNFREEZE_LAST_LAYERS,
        total_layers
    )

    for layer in base_model.layers[
        -layers_to_unfreeze:
    ]:

        layer.trainable = True

    # BatchNormalization layers should remain frozen
    # during small-data fine-tuning for stability.
    for layer in base_model.layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

    trainable_count = sum(
        layer.trainable
        for layer in base_model.layers
    )

    print(
        f"Layers selected for fine-tuning: "
        f"{layers_to_unfreeze}"
    )

    print(
        f"Actually trainable base layers: "
        f"{trainable_count}"
    )

    # Recompile with small learning rate
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
    print("LEAFLENS CNN - FINE-TUNING")
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
        f"Total images: "
        f"{len(full_dataset)}"
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    labels = np.asarray(
        full_dataset["label"],
        dtype=np.int32
    )

    (
        train_indices,
        val_indices,
        test_indices,
        class_names
    ) = load_split()

    print("\n" + "=" * 70)
    print("USING EXISTING DATASET SPLIT")
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

    print(
        "\nThe test set remains untouched."
    )

    # --------------------------------------------------------
    # Load existing best model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Best CNN model not found:\n"
            f"{MODEL_PATH}"
        )

    print("\n" + "=" * 70)
    print("LOADING EXISTING BEST CNN")
    print("=" * 70)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"\nLoaded model:"
    )

    print(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Configure fine-tuning
    # --------------------------------------------------------

    model = configure_fine_tuning(
        model
    )

    print("\n" + "=" * 70)
    print("MODEL AFTER FINE-TUNING SETUP")
    print("=" * 70)

    trainable_params = np.sum(
        [
            np.prod(
                variable.shape
            )
            for variable in model.trainable_variables
        ]
    )

    non_trainable_params = np.sum(
        [
            np.prod(
                variable.shape
            )
            for variable in model.non_trainable_variables
        ]
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_params:,}"
    )

    # --------------------------------------------------------
    # Create generators
    # --------------------------------------------------------

    print(
        "\nCreating training generators..."
    )

    train_generator = LeafDataGenerator(
        dataset=full_dataset,
        indices=train_indices,
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_generator = LeafDataGenerator(
        dataset=full_dataset,
        indices=val_indices,
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        f"Training batches: "
        f"{len(train_generator)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_generator)}"
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=FINE_TUNED_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
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
    print("STARTING FINE-TUNING")
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
        f"Unfrozen layers: "
        f"{UNFREEZE_LAST_LAYERS}"
    )

    print(
        "\nStarting from the existing "
        "87.73% test-performance model."
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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    best_epoch = int(
        np.argmax(
            history.history[
                "val_accuracy"
            ]
        )
    )

    best_train_accuracy = (
        history.history[
            "accuracy"
        ][best_epoch]
    )

    best_val_accuracy = (
        history.history[
            "val_accuracy"
        ][best_epoch]
    )

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Best epoch: "
        f"{best_epoch + 1}"
    )

    print(
        f"Training accuracy: "
        f"{best_train_accuracy:.2%}"
    )

    print(
        f"Validation accuracy: "
        f"{best_val_accuracy:.2%}"
    )

    print(
        f"\nBest model saved to:"
    )

    print(
        FINE_TUNED_MODEL_PATH
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        OUTPUT_DIR,
        "mobilenetv3_finetuning_history.npz"
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

    print(
        "\nTraining history saved:"
    )

    print(
        history_path
    )

    print("\n" + "=" * 70)
    print("FINE-TUNING PHASE FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
