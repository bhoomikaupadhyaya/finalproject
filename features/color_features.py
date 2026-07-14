import cv2
import numpy as np


# ----------------------------------------
# Mean RGB
# ----------------------------------------

def rgb_features(image):

    b, g, r = cv2.split(image)

    return {
        "mean_red": np.mean(r),
        "mean_green": np.mean(g),
        "mean_blue": np.mean(b)
    }


# ----------------------------------------
# Mean HSV
# ----------------------------------------

def hsv_features(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    return {

        "mean_hue": np.mean(h),

        "mean_saturation": np.mean(s),

        "mean_value": np.mean(v)

    }


# ----------------------------------------
# Mean LAB
# ----------------------------------------

def lab_features(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    return {

        "mean_L": np.mean(l),

        "mean_A": np.mean(a),

        "mean_B": np.mean(b)

    }


# ----------------------------------------
# All Color Features
# ----------------------------------------

def extract_color_features(image):

    features = {}

    features.update(rgb_features(image))

    features.update(hsv_features(image))

    features.update(lab_features(image))

    return features