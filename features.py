import cv2  # image read, resize, color conversion
import numpy as np  # arrays and numeric ops
from skimage.feature import hog, graycomatrix, graycoprops  # HOG and GLCM
from PIL import Image  # optional PIL input type

IMG_SIZE = 128  # width/height for all resized images
FEATURE_LEN = 368  # total feature count: colour + GLCM + HOG

GLCM_DISTANCES = [1, 3]  # neighbour offsets for co-occurrence pairs
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # four directions in radians
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]  # stats from GLCM

HOG_PIXELS_PER_CELL = (32, 32)  # cell size on 128×128 grid
HOG_CELLS_PER_BLOCK = (2, 2)  # block size for local contrast norm
HOG_ORIENTATIONS = 9  # gradient orientation bins


def extract_features(image_input):
    if isinstance(image_input, str):  # path string
        bgr = cv2.imread(image_input)  # load BGR from disk
        if bgr is None:  # missing or unreadable file
            raise ValueError(f"Could not read image: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # convert to RGB

    elif isinstance(image_input, Image.Image):  # PIL image
        rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)  # force RGB uint8 array

    elif isinstance(image_input, np.ndarray):  # already an array
        rgb = image_input.astype(np.uint8)  # ensure 8-bit unsigned
    else:  # unsupported type
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)  # fixed spatial size

    colour_feats = _colour_features(rgb)  # 12-d colour stats
    texture_feats = _glcm_features(rgb)  # 32-d GLCM texture
    hog_feats = _hog_features(rgb)  # 324-d HOG shape/edges

    feature_vector = np.concatenate([colour_feats, texture_feats, hog_feats])  # single 368-d vector
    return feature_vector.astype(np.float32)  # float32 for sklearn


def _colour_features(rgb):
    rgb_float = rgb.astype(np.float32) / 255.0  # RGB in [0, 1]
    rgb_feats = []  # collect per-channel stats
    for ch in range(3):  # R, G, B
        rgb_feats.append(rgb_float[:, :, ch].mean())  # channel mean
        rgb_feats.append(rgb_float[:, :, ch].std())  # channel std

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)  # HSV in OpenCV ranges
    hsv[:, :, 0] /= 179.0  # scale H to ~[0, 1]
    hsv[:, :, 1] /= 255.0  # scale S to [0, 1]
    hsv[:, :, 2] /= 255.0  # scale V to [0, 1]
    hsv_feats = []  # collect HSV stats
    for ch in range(3):  # H, S, V
        hsv_feats.append(hsv[:, :, ch].mean())  # channel mean
        hsv_feats.append(hsv[:, :, ch].std())  # channel std

    return np.array(rgb_feats + hsv_feats, dtype=np.float32)  # length 12


def _glcm_features(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # single channel for texture
    grey_reduced = (grey // 8).astype(np.uint8)  # quantize 256→32 grey levels

    glcm = graycomatrix(  # build normalized symmetric GLCM
        grey_reduced,  # 8-bit labels 0..31
        distances=GLCM_DISTANCES,  # pair offsets
        angles=GLCM_ANGLES,  # pair directions
        levels=32,  # must match quantization
        symmetric=True,  # count (i,j) and (j,i)
        normed=True,  # probabilities not raw counts
    )

    feats = []  # flatten all property×distance×angle values
    for prop in GLCM_PROPS:  # each scalar texture measure
        values = graycoprops(glcm, prop)  # shape (2 distances, 4 angles)
        feats.extend(values.flatten())  # 8 numbers per property

    return np.array(feats, dtype=np.float32)  # length 32


def _hog_features(rgb):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # HOG on intensity

    feats = hog(  # oriented gradient histograms
        grey,  # 128×128 grey image
        orientations=HOG_ORIENTATIONS,  # bin count
        pixels_per_cell=HOG_PIXELS_PER_CELL,  # cell grid
        cells_per_block=HOG_CELLS_PER_BLOCK,  # block for normalization
        block_norm="L2-Hys",  # normalize then clip renorm
        feature_vector=True,  # 1-D output
    )
    return feats.astype(np.float32)  # length 324


def validate_feature_length():
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))  # blank RGB image
    vec = extract_features(dummy)  # run full pipeline
    assert vec.shape == (FEATURE_LEN,), (  # must match declared length
        f"Feature length mismatch: got {vec.shape[0]}, expected {FEATURE_LEN}"
    )
    return True  # checks passed


if __name__ == "__main__":  # run as script
    validate_feature_length()  # self-test
    print(f"Feature extraction OK — vector length: {FEATURE_LEN}")  # success message
