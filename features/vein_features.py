import cv2
import numpy as np


# ----------------------------------------
# Vein Features
# ----------------------------------------

def extract_vein_features(image):

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image

    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Total Pixels
    total_pixels = binary.shape[0] * binary.shape[1]

    # Vein Pixels
    vein_pixels = cv2.countNonZero(binary)

    # Vein Density
    vein_density = vein_pixels / total_pixels

    # Connected Components
    num_labels, labels = cv2.connectedComponents(binary)

    # Vein Length (Approx.)
    vein_length = vein_pixels

    # Branch Count
    branch_count = num_labels - 1

    return {

        "vein_density": vein_density,

        "vein_pixels": vein_pixels,

        "vein_length": vein_length,

        "branch_count": branch_count

    }


# ----------------------------------------
# Testing
# ----------------------------------------

if __name__ == "__main__":

    image = cv2.imread(
        "../output/skeleton/healthy/leaf1.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    if image is not None:

        features = extract_vein_features(image)

        for key, value in features.items():

            print(f"{key} : {value}")