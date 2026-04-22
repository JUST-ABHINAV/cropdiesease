"""
features.py  —  CropScan Feature Extraction Module
====================================================
Converts a raw leaf image into a fixed-length numerical feature vector
that scikit-learn classifiers can work with.

Feature groups:
  1. Colour features  — mean + std of each R, G, B channel (6 values)
                        mean + std of each H, S, V channel (6 values)
                        → 12 features total

  2. Texture features — GLCM (Grey-Level Co-occurrence Matrix)
                        properties: contrast, homogeneity, energy, correlation
                        at 2 distances × 4 angles = 32 values
                        → 32 features total

  3. Edge / shape     — HOG (Histogram of Oriented Gradients)
                        captures where edges and gradients are in the leaf
                        → 324 features total

Grand total: 12 + 32 + 324 = 368 features per image
"""

import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE     = 128          # resize all images to 128 × 128 pixels
FEATURE_LEN  = 368          # must match: 12 + 32 + 324

# GLCM settings
GLCM_DISTANCES = [1, 3]     # pixel distances to consider
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]   # 4 directions
GLCM_PROPS     = ["contrast", "homogeneity", "energy", "correlation"]

# HOG settings (on 128×128 image)
HOG_PIXELS_PER_CELL = (32, 32)   # 128/32 = 4 cells per side
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9
# blocks = (4-2+1)×(4-2+1) = 9  →  9 × 2×2 × 9 = 324 values


def extract_features(image_input):
    """
    Convert a single image to a 368-dimensional feature vector.

    Parameters
    ----------
    image_input : str, PIL.Image, or np.ndarray
        Accepts a file path, a PIL Image, or a numpy array (uint8 RGB).

    Returns
    -------
    np.ndarray  shape (368,)  dtype float32
    """

    # ── Step 1: Load & standardise to numpy RGB uint8 ──────────────────────────
    if isinstance(image_input, str):
        # File path — read with OpenCV (returns BGR), convert to RGB
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif isinstance(image_input, Image.Image):
        # PIL Image
        rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)

    elif isinstance(image_input, np.ndarray):
        rgb = image_input.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    # ── Step 2: Resize to fixed size ───────────────────────────────────────────
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # ── Feature Group 1: Colour features (12 values) ──────────────────────────
    colour_feats = _colour_features(rgb)    # 12

    # ── Feature Group 2: Texture features (32 values) ─────────────────────────
    texture_feats = _glcm_features(rgb)     # 32

    # ── Feature Group 3: Edge / HOG features (324 values) ─────────────────────
    hog_feats = _hog_features(rgb)          # 324

    # ── Concatenate all into one vector ───────────────────────────────────────
    feature_vector = np.concatenate([colour_feats, texture_feats, hog_feats])
    return feature_vector.astype(np.float32)


# ── Private helpers ────────────────────────────────────────────────────────────

def _colour_features(rgb):
    """
    Compute mean and std of each channel in RGB and HSV colour spaces.
    Returns 12 values: [R_mean, R_std, G_mean, G_std, B_mean, B_std,
                        H_mean, H_std, S_mean, S_std, V_mean, V_std]
    """
    # RGB stats
    rgb_float = rgb.astype(np.float32) / 255.0
    rgb_feats = []
    for ch in range(3):                          # R, G, B
        rgb_feats.append(rgb_float[:, :, ch].mean())
        rgb_feats.append(rgb_float[:, :, ch].std())

    # HSV stats (OpenCV HSV: H in 0–179, S and V in 0–255)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0   # normalise H to 0–1
    hsv[:, :, 1] /= 255.0   # normalise S to 0–1
    hsv[:, :, 2] /= 255.0   # normalise V to 0–1
    hsv_feats = []
    for ch in range(3):                          # H, S, V
        hsv_feats.append(hsv[:, :, ch].mean())
        hsv_feats.append(hsv[:, :, ch].std())

    return np.array(rgb_feats + hsv_feats, dtype=np.float32)   # 12 values


def _glcm_features(rgb):
    """
    Compute GLCM (Grey-Level Co-occurrence Matrix) texture features.

    GLCM tells us how often pairs of pixels with specific values appear
    at a given distance and angle — this captures leaf surface texture.

    Returns 32 values: 4 properties × 2 distances × 4 angles
    """
    # Convert to greyscale (GLCM works on single-channel images)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Reduce to 32 grey levels to keep GLCM computation fast
    grey_reduced = (grey // 8).astype(np.uint8)   # 256 → 32 levels

    # Compute GLCM matrix
    glcm = graycomatrix(
        grey_reduced,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=32,
        symmetric=True,
        normed=True,
    )

    # Extract 4 properties from the GLCM
    # Each property returns shape (len(distances), len(angles)) = (2, 4)
    feats = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)   # shape (2, 4)
        feats.extend(values.flatten())     # 2 × 4 = 8 values per property

    return np.array(feats, dtype=np.float32)   # 4 × 8 = 32 values


def _hog_features(rgb):
    """
    Compute HOG (Histogram of Oriented Gradients) features.

    HOG captures the distribution of edge directions in the image.
    It is particularly good at detecting the veins, spots, and
    structural patterns in diseased leaves.

    Returns 324 values.
    """
    # HOG works on greyscale
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    feats = hog(
        grey,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)   # 324 values


def validate_feature_length():
    """Quick sanity check — verify the feature vector is exactly FEATURE_LEN."""
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    vec = extract_features(dummy)
    assert vec.shape == (FEATURE_LEN,), (
        f"Feature length mismatch: got {vec.shape[0]}, expected {FEATURE_LEN}"
    )
    return True


if __name__ == "__main__":
    validate_feature_length()
    print(f"Feature extraction OK — vector length: {FEATURE_LEN}")
