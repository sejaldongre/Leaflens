import os
import time

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "data/processed/dataset_split.npz"
RESULT_FILE = "data/processed/random_forest_tuning.npz"

RANDOM_STATE = 42
PCA_COMPONENTS = 30


def evaluate_configuration(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val
):
    print("\n" + "-" * 70)
    print(f"Configuration: {name}")
    print("-" * 70)

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    elapsed = time.time() - start_time

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted"
    )

    print(
        f"Accuracy   : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1   : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Time       : "
        f"{elapsed:.2f} seconds"
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "time": elapsed
    }


def main():

    print("=" * 70)
    print("LEAFLENS RANDOM FOREST TUNING")
    print("=" * 70)

    # --------------------------------------------------
    # Load original split
    # --------------------------------------------------

    data = np.load(
        INPUT_FILE,
        allow_pickle=True
    )

    X_train = data["X_train"]
    X_val = data["X_val"]

    y_train = data["y_train"]
    y_val = data["y_val"]

    print("\nOriginal features:")
    print(
        f"Training   : {X_train.shape}"
    )
    print(
        f"Validation : {X_val.shape}"
    )

    # --------------------------------------------------
    # StandardScaler
    # --------------------------------------------------

    print("\nApplying StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # --------------------------------------------------
    # PCA = 30
    # --------------------------------------------------

    print(
        f"\nApplying PCA "
        f"with {PCA_COMPONENTS} components..."
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_val_pca = pca.transform(
        X_val_scaled
    )

    variance = np.sum(
        pca.explained_variance_ratio_
    )

    print(
        f"Variance retained: "
        f"{variance * 100:.2f}%"
    )

    print(
        f"Training shape: "
        f"{X_train_pca.shape}"
    )

    print(
        f"Validation shape: "
        f"{X_val_pca.shape}"
    )

    # --------------------------------------------------
    # Random Forest configurations
    # --------------------------------------------------

    configurations = [

        (
            "RF_200_trees",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_trees",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_500_trees",
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_depth_20",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_depth_30",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_min_leaf_2",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_min_leaf_3",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_sqrt",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "RF_300_log2",
            RandomForestClassifier(
                n_estimators=300,
                max_features="log2",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

    ]

    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------

    results = []

    for name, model in configurations:

        result = evaluate_configuration(
            name,
            model,
            X_train_pca,
            y_train,
            X_val_pca,
            y_val
        )

        results.append(
            result
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n" + "=" * 90)
    print("RANDOM FOREST TUNING RESULTS")
    print("=" * 90)

    print(
        f"\n{'Configuration':<25}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
        f"{'Weighted F1':<15}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['name']:<25}"
            f"{result['accuracy'] * 100:>8.2f}%       "
            f"{result['macro_f1']:>8.4f}       "
            f"{result['weighted_f1']:>8.4f}"
        )

    # --------------------------------------------------
    # Select best configuration
    # --------------------------------------------------

    best_result = max(
        results,
        key=lambda item: item["macro_f1"]
    )

    print("\n" + "=" * 70)
    print("BEST RANDOM FOREST CONFIGURATION")
    print("=" * 70)

    print(
        f"\nConfiguration : "
        f"{best_result['name']}"
    )

    print(
        f"Accuracy      : "
        f"{best_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1      : "
        f"{best_result['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1   : "
        f"{best_result['weighted_f1']:.4f}"
    )

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    os.makedirs(
        "data/processed/models",
        exist_ok=True
    )

    best_model_file = (
        "data/processed/models/"
        "best_random_forest_validation.pkl"
    )

    joblib.dump(
        best_result["model"],
        best_model_file
    )

    # Save scaler and PCA used for this model

    joblib.dump(
        scaler,
        "data/processed/models/"
        "rf_scaler.pkl"
    )

    joblib.dump(
        pca,
        "data/processed/models/"
        "rf_pca.pkl"
    )

    # --------------------------------------------------
    # Save tuning results
    # --------------------------------------------------

    np.savez_compressed(
        RESULT_FILE,
        names=np.array([
            result["name"]
            for result in results
        ]),
        accuracy=np.array([
            result["accuracy"]
            for result in results
        ]),
        macro_f1=np.array([
            result["macro_f1"]
            for result in results
        ]),
        weighted_f1=np.array([
            result["weighted_f1"]
            for result in results
        ]),
        training_time=np.array([
            result["time"]
            for result in results
        ])
    )

    print("\nSaved best model to:")
    print(
        best_model_file
    )

    print("\nSaved scaler to:")
    print(
        "data/processed/models/rf_scaler.pkl"
    )

    print("\nSaved PCA to:")
    print(
        "data/processed/models/rf_pca.pkl"
    )

    print("\nSaved tuning results to:")
    print(
        RESULT_FILE
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
