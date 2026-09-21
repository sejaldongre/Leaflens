import os
import time

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "data/processed/dataset_split.npz"
MODEL_DIR = "data/processed/models"

RANDOM_STATE = 42

# Best configuration from validation experiment
PCA_COMPONENTS = 40
N_ESTIMATORS = 400
MAX_DEPTH = 20


def main():
    print("=" * 70)
    print("LEAFLENS FINAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load train / validation / test split
    # --------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = np.load(
        INPUT_FILE,
        allow_pickle=True
    )

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    class_names = data["class_names"]

    print("\nOriginal data:")
    print(
        f"Training   : {X_train.shape}"
    )
    print(
        f"Validation : {X_val.shape}"
    )
    print(
        f"Test       : {X_test.shape}"
    )

    # --------------------------------------------------
    # Combine training + validation
    #
    # Validation was used to select the model
    # configuration.
    #
    # Selected configuration:
    # PCA = 40
    # Random Forest = 400 trees
    # Max depth = 20
    #
    # The test set remains completely untouched.
    # --------------------------------------------------

    X_train_final = np.concatenate(
        [
            X_train,
            X_val
        ],
        axis=0
    )

    y_train_final = np.concatenate(
        [
            y_train,
            y_val
        ],
        axis=0
    )

    print("\nFinal training data:")
    print(
        f"Training + Validation : "
        f"{X_train_final.shape}"
    )

    print(
        f"Final Test            : "
        f"{X_test.shape}"
    )

    # --------------------------------------------------
    # StandardScaler
    #
    # Fit ONLY on final training data.
    # --------------------------------------------------

    print("\nApplying StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_final
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("Scaling complete.")

    # --------------------------------------------------
    # PCA
    # --------------------------------------------------

    print(
        f"\nApplying PCA with "
        f"{PCA_COMPONENTS} components..."
    )

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    explained_variance = np.sum(
        pca.explained_variance_ratio_
    )

    print(
        f"Variance retained: "
        f"{explained_variance * 100:.2f}%"
    )

    print(
        f"Training shape: "
        f"{X_train_pca.shape}"
    )

    print(
        f"Test shape    : "
        f"{X_test_pca.shape}"
    )

    # --------------------------------------------------
    # Train final Random Forest
    # --------------------------------------------------

    print("\nTraining final Random Forest...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    training_start = time.time()

    model.fit(
        X_train_pca,
        y_train_final
    )

    training_time = (
        time.time()
        - training_start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )

    # --------------------------------------------------
    # Final prediction on untouched test set
    # --------------------------------------------------

    print(
        "\nRunning final prediction "
        "on test set..."
    )

    prediction_start = time.time()

    y_pred = model.predict(
        X_test_pca
    )

    prediction_time = (
        time.time()
        - prediction_start
    )

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nTest Accuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Prediction time  : "
        f"{prediction_time:.4f} seconds"
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    confusion = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "Confusion matrix shape: "
        f"{confusion.shape}"
    )

    # --------------------------------------------------
    # Save final model artifacts
    # --------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_file = os.path.join(
        MODEL_DIR,
        "leaflens_random_forest.pkl"
    )

    scaler_file = os.path.join(
        MODEL_DIR,
        "leaflens_scaler.pkl"
    )

    pca_file = os.path.join(
        MODEL_DIR,
        "leaflens_pca.pkl"
    )

    report_file = os.path.join(
        MODEL_DIR,
        "classification_report.txt"
    )

    confusion_file = os.path.join(
        MODEL_DIR,
        "confusion_matrix.npy"
    )

    metrics_file = os.path.join(
        MODEL_DIR,
        "final_metrics.npz"
    )

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    joblib.dump(
        pca,
        pca_file
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    np.save(
        confusion_file,
        confusion
    )

    np.savez_compressed(
        metrics_file,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        training_time=training_time,
        prediction_time=prediction_time,
        pca_components=PCA_COMPONENTS,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_names=class_names
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\nSaved final model:")
    print(model_file)

    print("\nSaved scaler:")
    print(scaler_file)

    print("\nSaved PCA:")
    print(pca_file)

    print("\nSaved classification report:")
    print(report_file)

    print("\nSaved confusion matrix:")
    print(confusion_file)

    print("\nSaved final metrics:")
    print(metrics_file)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
