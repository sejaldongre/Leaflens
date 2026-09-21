import numpy as np


FEATURE_FILE = "data/processed/features_full.npz"


def main():
    print("=" * 60)
    print("CHECKING LEAFLENS FEATURE DATA")
    print("=" * 60)

    data = np.load(
        FEATURE_FILE,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]
    class_names = data["class_names"]

    print(f"\nFeature matrix shape : {X.shape}")
    print(f"Labels shape         : {y.shape}")
    print(f"Number of classes    : {len(class_names)}")
    print(f"Feature data type    : {X.dtype}")
    print(f"Label data type      : {y.dtype}")

    print("\nClass names:")
    for index, name in enumerate(class_names):
        print(f"{index:2d} -> {name}")

    print("\nLabel range:")
    print(f"Minimum label : {y.min()}")
    print(f"Maximum label : {y.max()}")

    print("\nFeature statistics:")
    print(f"Minimum feature value : {X.min():.4f}")
    print(f"Maximum feature value : {X.max():.4f}")
    print(f"Mean feature value    : {X.mean():.4f}")

    print("\nMissing / invalid values:")
    print(f"NaN values       : {np.isnan(X).sum()}")
    print(f"Infinite values  : {np.isinf(X).sum()}")

    print("\n" + "=" * 60)
    print("FEATURE DATA CHECK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
