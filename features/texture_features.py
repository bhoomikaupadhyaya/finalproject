import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ----------------------------------------
# GLCM Features
# ----------------------------------------

def glcm_features(gray):

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = {

        "contrast": graycoprops(glcm, "contrast")[0, 0],

        "dissimilarity": graycoprops(glcm, "dissimilarity")[0, 0],

        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],

        "energy": graycoprops(glcm, "energy")[0, 0],

        "correlation": graycoprops(glcm, "correlation")[0, 0],

        "ASM": graycoprops(glcm, "ASM")[0, 0]

    }

    return features


# ----------------------------------------
# LBP Features
# ----------------------------------------

def lbp_features(gray):

    radius = 2

    points = radius * 8

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, points + 3),
        range=(0, points + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)

    features = {}

    for i, value in enumerate(hist):

        features[f"lbp_{i}"] = value

    return features


# ----------------------------------------
# Entropy
# ----------------------------------------

def entropy(gray):

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist / hist.sum()

    hist = hist.flatten()

    ent = -np.sum(
        hist * np.log2(hist + 1e-10)
    )

    return {

        "entropy": ent

    }


# ----------------------------------------
# Extract Texture Features
# ----------------------------------------

def extract_texture_features(image):

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image

    features = {}

    features.update(glcm_features(gray))

    features.update(lbp_features(gray))

    features.update(entropy(gray))

    return features


# ----------------------------------------
# Testing
# ----------------------------------------

if __name__ == "__main__":

    image = cv2.imread(
        "../output/morphology/healthy/leaf1.jpg"
    )

    if image is not None:

        features = extract_texture_features(image)

        for key, value in features.items():

            print(f"{key}: {value}")