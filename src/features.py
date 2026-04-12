import cv2
import numpy as np
import logging
from skimage.feature import hog
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

def preprocess_image(image_path: str, size: Tuple[int, int] = (64, 64)) -> Optional[np.ndarray]:
    """Load, resize, and convert an image to grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not load image: {image_path}")
        return None
    
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_hog_features(image: np.ndarray, config: dict) -> np.ndarray:
    """Extract HOG features from a grayscale image."""
    hog_config = config['HOG']
    features = hog(
        image,
        orientations=hog_config['ORIENTATIONS'],
        pixels_per_cell=tuple(hog_config['PIXELS_PER_CELL']),
        cells_per_block=tuple(hog_config['CELLS_PER_BLOCK']),
        visualize=hog_config['VISUALIZE']
    )
    return features
