import os
import time

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


INPUT_FILE = "data/processed/dataset_split.npz"
RESULT_FILE = "data/processed/pca_comparison.npz"

RANDOM_STATE = 42


def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):
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

    return accuracy, macro_f1, elapsed


def main():
    print("=" * 70)
    print("LEAFLENS PCA DIMENSION COMPARISON")
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

    print("\nOriginal data:")
    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")

    # --------------------------------------------------
    # Scale features
    #
    # Fit ONLY on training data.
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
    # PCA configurations
    # --------------------------------------------------

    pca_components = [
        5,
        10,
        20,
        30,
        40,
        50,
        None
    ]

    results = []

    # --------------------------------------------------
    # Experiment
    # --------------------------------------------------

    for n_components in pca_components:

        print("\n" + "=" * 70)

        if n_components is None:
            print("FEATURE SET: ALL 55 FEATURES")

            X_train_current = X_train_scaled
            X_val_current = X_val_scaled

            actual_components = X_train_current.shape[1]

        else:
            print(
                f"FEATURE SET: PCA "
                f"{n_components} COMPONENTS"
            )

            pca = PCA(
                n_components=n_components,
                random_state=RANDOM_STATE
            )

            X_train_current = pca.fit_transform(
                X_train_scaled
            )

            X_val_current = pca.transform(
                X_val_scaled
            )

            actual_components = (
                X_train_current.shape[1]
            )

            variance = np.sum(
                pca.explained_variance_ratio_
            )

            print(
                f"Variance retained: "
                f"{variance * 100:.2f}%"
            )

        print(
            f"Feature dimensions: "
            f"{actual_components}"
        )

        # --------------------------------------------------
        # Random Forest
        # --------------------------------------------------

        print("\nTraining Random Forest...")

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        )

        rf_accuracy, rf_f1, rf_time = evaluate_model(
            rf,
            X_train_current,
            y_train,
            X_val_current,
            y_val
        )

        print(
            f"Random Forest Accuracy : "
            f"{rf_accuracy * 100:.2f}%"
        )

        print(
            f"Random Forest Macro F1 : "
            f"{rf_f1:.4f}"
        )

        # --------------------------------------------------
        # SVM
        # --------------------------------------------------

        print("\nTraining SVM...")

        svm = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=RANDOM_STATE
        )

        svm_accuracy, svm_f1, svm_time = evaluate_model(
            svm,
            X_train_current,
            y_train,
            X_val_current,
            y_val
        )

        print(
            f"SVM Accuracy           : "
            f"{svm_accuracy * 100:.2f}%"
        )

        print(
            f"SVM Macro F1           : "
            f"{svm_f1:.4f}"
        )

        results.append({
            "components": actual_components,
            "rf_accuracy": rf_accuracy,
            "rf_f1": rf_f1,
            "rf_time": rf_time,
            "svm_accuracy": svm_accuracy,
            "svm_f1": svm_f1,
            "svm_time": svm_time
        })

    # --------------------------------------------------
    # Print final comparison
    # --------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL PCA COMPARISON")
    print("=" * 90)

    print(
        f"\n{'Features':<12}"
        f"{'RF Accuracy':<16}"
        f"{'RF F1':<12}"
        f"{'SVM Accuracy':<16}"
        f"{'SVM F1':<12}"
    )

    print("-" * 68)

    for result in results:

        print(
            f"{result['components']:<12}"
            f"{result['rf_accuracy'] * 100:>8.2f}%       "
            f"{result['rf_f1']:>8.4f}    "
            f"{result['svm_accuracy'] * 100:>8.2f}%       "
            f"{result['svm_f1']:>8.4f}"
        )

    # --------------------------------------------------
    # Find best configuration
    # --------------------------------------------------

    best_rf = max(
        results,
        key=lambda item: item["rf_f1"]
    )

    best_svm = max(
        results,
        key=lambda item: item["svm_f1"]
    )

    print("\n" + "=" * 70)
    print("BEST RANDOM FOREST CONFIGURATION")
    print("=" * 70)

    print(
        f"Features : "
        f"{best_rf['components']}"
    )

    print(
        f"Accuracy : "
        f"{best_rf['rf_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{best_rf['rf_f1']:.4f}"
    )

    print("\n" + "=" * 70)
    print("BEST SVM CONFIGURATION")
    print("=" * 70)

    print(
        f"Features : "
        f"{best_svm['components']}"
    )

    print(
        f"Accuracy : "
        f"{best_svm['svm_accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{best_svm['svm_f1']:.4f}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    np.savez_compressed(
        RESULT_FILE,
        components=np.array([
            r["components"]
            for r in results
        ]),
        rf_accuracy=np.array([
            r["rf_accuracy"]
            for r in results
        ]),
        rf_f1=np.array([
            r["rf_f1"]
            for r in results
        ]),
        svm_accuracy=np.array([
            r["svm_accuracy"]
            for r in results
        ]),
        svm_f1=np.array([
            r["svm_f1"]
            for r in results
        ])
    )

    print("\nSaved comparison results to:")
    print(RESULT_FILE)

    print("\n" + "=" * 70)
    print("PCA COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
