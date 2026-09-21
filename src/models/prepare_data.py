import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


INPUT_FILE = "data/processed/dataset_split.npz"
OUTPUT_FILE = "data/processed/dataset_prepared.npz"

RANDOM_STATE = 42

# Keep enough components to retain 95% of the variance
PCA_VARIANCE = 0.95


def main():
    print("=" * 60)
    print("PREPARING LEAFLENS FEATURES")
    print("=" * 60)

    # --------------------------------------------------
    # Load train / validation / test data
    # --------------------------------------------------

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

    print("\nOriginal feature shapes:")
    print(f"Training   : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    # --------------------------------------------------
    # StandardScaler
    #
    # IMPORTANT:
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

    X_test_scaled = scaler.transform(
        X_test
    )

    print("Scaling complete.")

    # --------------------------------------------------
    # PCA
    #
    # Fit ONLY on training data.
    # Retain 95% variance.
    # --------------------------------------------------

    print(
        f"\nApplying PCA "
        f"(variance retained = {PCA_VARIANCE:.0%})..."
    )

    pca = PCA(
        n_components=PCA_VARIANCE,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_val_pca = pca.transform(
        X_val_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    explained_variance = (
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    print("\nPCA complete.")

    print(
        f"Original features      : "
        f"{X_train.shape[1]}"
    )

    print(
        f"PCA components         : "
        f"{X_train_pca.shape[1]}"
    )

    print(
        f"Explained variance     : "
        f"{explained_variance:.4f}"
    )

    print(
        f"Variance retained      : "
        f"{explained_variance * 100:.2f}%"
    )

    # --------------------------------------------------
    # Final shapes
    # --------------------------------------------------

    print("\nPrepared feature shapes:")
    print(
        f"Training   : "
        f"{X_train_pca.shape}"
    )

    print(
        f"Validation : "
        f"{X_val_pca.shape}"
    )

    print(
        f"Test       : "
        f"{X_test_pca.shape}"
    )

    # --------------------------------------------------
    # Save processed data
    # --------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train_pca.astype(
            np.float32
        ),
        X_val=X_val_pca.astype(
            np.float32
        ),
        X_test=X_test_pca.astype(
            np.float32
        ),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        class_names=class_names
    )

    # --------------------------------------------------
    # Save scaler and PCA for inference
    # --------------------------------------------------

    import joblib

    scaler_file = (
        "data/processed/scaler.pkl"
    )

    pca_file = (
        "data/processed/pca.pkl"
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    joblib.dump(
        pca,
        pca_file
    )

    print("\nSaved files:")
    print(
        f"Prepared dataset : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Scaler           : "
        f"{scaler_file}"
    )

    print(
        f"PCA              : "
        f"{pca_file}"
    )

    print("\n" + "=" * 60)
    print("FEATURE PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
