import cv2
import numpy as np


# ----------------------------------------
# Shape Features
# ----------------------------------------

def extract_shape_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        return {

            "leaf_area": 0,

            "perimeter": 0,

            "aspect_ratio": 0,

            "extent": 0,

            "solidity": 0,

            "circularity": 0

        }

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h if h != 0 else 0

    bounding_area = w * h

    extent = (
        area / bounding_area
        if bounding_area != 0
        else 0
    )

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = (
        area / hull_area
        if hull_area != 0
        else 0
    )

    circularity = (
        (4 * np.pi * area) /
        (perimeter * perimeter)
        if perimeter != 0
        else 0
    )

    return {

        "leaf_area": area,

        "perimeter": perimeter,

        "aspect_ratio": aspect_ratio,

        "extent": extent,

        "solidity": solidity,

        "circularity": circularity

    }


# ----------------------------------------
# Testing
# ----------------------------------------

if __name__ == "__main__":

    image = cv2.imread(
        "../output/morphology/healthy/leaf1.jpg"
    )

    if image is not None:

        features = extract_shape_features(image)

        for key, value in features.items():

            print(f"{key} : {value}")