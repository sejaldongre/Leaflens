import os
import time

import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


INPUT_FILE = "data/processed/dataset_prepared.npz"
OUTPUT_DIR = "data/processed/models"

RANDOM_STATE = 42


def evaluate_model(
    model,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val
):
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    predictions = model.predict(
        X_val
    )

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
        f"Training time : "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Validation Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Validation Macro F1 : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Validation Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    return {
        "name": model_name,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time": training_time
    }


def main():
    print("=" * 60)
    print("LEAFLENS MODEL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------
    # Load prepared data
    # --------------------------------------------------

    data = np.load(
        INPUT_FILE,
        allow_pickle=True
    )

    X_train = data["X_train"]
    X_val = data["X_val"]

    y_train = data["y_train"]
    y_val = data["y_val"]

    class_names = data["class_names"]

    print("\nDataset:")
    print(
        f"Training features : "
        f"{X_train.shape}"
    )

    print(
        f"Validation features : "
        f"{X_val.shape}"
    )

    print(
        f"Number of classes : "
        f"{len(class_names)}"
    )

    # --------------------------------------------------
    # Define models
    # --------------------------------------------------

    models = [
        (
            "SVM",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                random_state=RANDOM_STATE
            )
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ),

        (
            "KNN",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1
            )
        ),

        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )
    ]

    # --------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------

    results = []

    for model_name, model in models:
        result = evaluate_model(
            model,
            model_name,
            X_train,
            y_train,
            X_val,
            y_val
        )

        results.append(
            result
        )

    # --------------------------------------------------
    # Comparison table
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Model':<22}"
        f"{'Accuracy':<15}"
        f"{'Macro F1':<15}"
        f"{'Weighted F1':<15}"
    )

    print("-" * 67)

    for result in results:
        print(
            f"{result['name']:<22}"
            f"{result['accuracy'] * 100:>8.2f}%       "
            f"{result['macro_f1']:>8.4f}       "
            f"{result['weighted_f1']:>8.4f}"
        )

    # --------------------------------------------------
    # Select model using Macro F1
    # --------------------------------------------------

    best_result = max(
        results,
        key=lambda item: item["macro_f1"]
    )

    print("\n" + "=" * 60)
    print("BEST MODEL ON VALIDATION SET")
    print("=" * 60)

    print(
        f"\nModel      : "
        f"{best_result['name']}"
    )

    print(
        f"Accuracy   : "
        f"{best_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1   : "
        f"{best_result['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{best_result['weighted_f1']:.4f}"
    )

    # --------------------------------------------------
    # Save comparison results
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    comparison_results = []

    for result in results:
        comparison_results.append({
            "model": result["name"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "training_time": result["training_time"]
        })

    np.savez_compressed(
        "data/processed/model_comparison.npz",
        model_names=np.array([
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
            result["training_time"]
            for result in results
        ])
    )

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------

    best_model_file = os.path.join(
        OUTPUT_DIR,
        "best_model_validation.pkl"
    )

    joblib.dump(
        best_result["model"],
        best_model_file
    )

    print("\nSaved model comparison to:")
    print(
        "data/processed/model_comparison.npz"
    )

    print("\nSaved best validation model to:")
    print(
        best_model_file
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
