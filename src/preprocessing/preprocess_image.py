import cv2
import numpy as np


# Size used for ML feature extraction.
# Keeping this relatively small makes training and inference faster.
IMAGE_SIZE = (256, 256)


def resize_image(image):
    """
    Resize an RGB image to a fixed size.
    """
    return cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def create_initial_mask(image):
    """
    Create an initial foreground mask using color information.

    This mask is only used to guide GrabCut.
    It is intentionally permissive so that useful parts of the
    leaf are not removed too early.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]

    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    # Green emphasis
    excess_green = 2 * g - r - b

    saturation_mask = saturation > 35
    green_mask = excess_green > 5

    combined = np.logical_or(
        saturation_mask,
        green_mask
    )

    mask = np.where(
        combined,
        cv2.GC_PR_FGD,
        cv2.GC_PR_BGD
    ).astype(np.uint8)

    # Border is treated as definite background.
    border = 5

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    return mask


def grabcut_segmentation(image):
    """
    Refine the initial foreground estimate using GrabCut.
    """

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    mask = create_initial_mask(image)

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image_bgr,
        mask,
        None,
        bgd_model,
        fgd_model,
        3,
        cv2.GC_INIT_WITH_MASK
    )

    # Keep definite and probable foreground.
    foreground = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Clean small holes/noise.
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Keep the largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground
    )

    if num_labels > 1:

        largest_component = (
            1 +
            np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )

        foreground = np.where(
            labels == largest_component,
            255,
            0
        ).astype(np.uint8)

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=foreground
    )

    return segmented, foreground


def preprocess_image(image):
    """
    Complete LeafLens preprocessing pipeline.

    Input:
        RGB image as a NumPy array.

    Returns:
        resized_image
        segmented_image
        segmentation_mask
        grayscale_image
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Ensure uint8 format.
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Resize before GrabCut to reduce computational cost.
    resized = resize_image(image)

    # Segment leaf.
    segmented, mask = grabcut_segmentation(resized)

    # Convert segmented image to grayscale.
    grayscale = cv2.cvtColor(
        segmented,
        cv2.COLOR_RGB2GRAY
    )

    return (
        resized,
        segmented,
        mask,
        grayscale
    )
