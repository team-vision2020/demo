"""
Image histograms for structured detection.

Author: Chunlok Lo, adapted by Mert
"""

import numpy as np
import skimage


def three_channel_histogram(images, bins=255):
    """
    Generates a 3 x (bins) array of binned R, G, B values from an image.
    """
    histograms = []
    for image in images:
        image = skimage.img_as_float(image)

        red = np.histogram(image[:, :, 0], bins=bins, range=(0, 1.0))[0]
        green = np.histogram(image[:, :, 0], bins=bins, range=(0, 1.0))[0]
        blue = np.histogram(image[:, :, 0], bins=bins, range=(0, 1.0))[0]
        histograms.append(np.hstack((red, green, blue)))

    return np.stack(histograms)
