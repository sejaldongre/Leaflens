import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

SPLIT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_split.npz"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load dataset split
# ============================================================

print("=" * 70)
print("LEAFLENS MODEL IMPROVEMENT EXPERIMENT")
print("=" * 70)

print("\nLoading dataset split...")

data = np.load(SPLIT_PATH)

X_train = data["X_train"]
X_val = data["X_val"]

y_train = data["y_train"]
y_val = data["y_val"]

print(f"Training features   : {X_train.shape}")
print(f"Validation features: {X_val.shape}")


# ============================================================
# Scale features
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# ============================================================
# PCA configurations
# ============================================================

pca_components = [20, 25, 30, 35, 40]


# ============================================================
# Random Forest configurations
# ============================================================

rf_configs = [
    {
        "name": "RF_200",
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "RF_600",
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400_depth20",
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400_depth30",
        "n_estimators": 400,
        "max_depth": 30,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400_leaf2",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400_leaf3",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
    },
    {
        "name": "RF_400_log2",
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "log2",
    },
]


# ============================================================
# Run experiments
# ============================================================

results = []

best_result = None


for n_components in pca_components:

    print("\n" + "-" * 70)
    print(f"PCA COMPONENTS: {n_components}")
    print("-" * 70)

    # Fit PCA ONLY on training data
    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    explained_variance = pca.explained_variance_ratio_.sum()

    print(
        f"Explained variance: "
        f"{explained_variance:.4%}"
    )

    print(
        f"Transformed shape: "
        f"{X_train_pca.shape}"
    )

    for config in rf_configs:

        print(
            f"\nTraining {config['name']}..."
        )

        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train_pca,
            y_train
        )

        predictions = model.predict(
            X_val_pca
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

        result = {
            "pca_components": n_components,
            "explained_variance": explained_variance,
            "model": config["name"],
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

        results.append(result)

        print(
            f"Accuracy    : {accuracy:.2%}"
        )

        print(
            f"Macro F1    : {macro_f1:.4f}"
        )

        print(
            f"Weighted F1 : {weighted_f1:.4f}"
        )

        # Select based on Macro F1
        if (
            best_result is None
            or macro_f1 > best_result["macro_f1"]
        ):
            best_result = result


# ============================================================
# Print all results
# ============================================================

print("\n" + "=" * 70)
print("ALL MODEL RESULTS")
print("=" * 70)

print(
    f"{'PCA':<6}"
    f"{'Model':<22}"
    f"{'Accuracy':<12}"
    f"{'Macro F1':<12}"
    f"{'Weighted F1':<12}"
)

print("-" * 70)

for result in sorted(
    results,
    key=lambda x: x["macro_f1"],
    reverse=True
):

    print(
        f"{result['pca_components']:<6}"
        f"{result['model']:<22}"
        f"{result['accuracy']:<12.2%}"
        f"{result['macro_f1']:<12.4f}"
        f"{result['weighted_f1']:<12.4f}"
    )


# ============================================================
# Best model
# ============================================================

print("\n" + "=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print(
    f"PCA components : "
    f"{best_result['pca_components']}"
)

print(
    f"Explained var.  : "
    f"{best_result['explained_variance']:.4%}"
)

print(
    f"Model           : "
    f"{best_result['model']}"
)

print(
    f"Validation acc. : "
    f"{best_result['accuracy']:.2%}"
)

print(
    f"Validation F1   : "
    f"{best_result['macro_f1']:.4f}"
)

print(
    f"Weighted F1     : "
    f"{best_result['weighted_f1']:.4f}"
)


# ============================================================
# Save experiment results
# ============================================================

np.savez(
    OUTPUT_DIR / "model_improvement_results.npz",
    results=np.array(
        [
            [
                r["pca_components"],
                r["explained_variance"],
                r["accuracy"],
                r["macro_f1"],
                r["weighted_f1"],
            ]
            for r in results
        ],
        dtype=float,
    ),
)

print("\nResults saved to:")

print(
    OUTPUT_DIR / "model_improvement_results.npz"
)

print("\nModel improvement experiment complete.")
