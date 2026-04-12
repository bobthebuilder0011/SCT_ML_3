import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from .features import preprocess_image, extract_hog_features

logger = logging.getLogger(__name__)

def process_single_image(img_path: Path, config: dict, is_test: bool) -> Optional[Tuple[np.ndarray, Optional[int], str]]:
    """Helper for parallel processing of images."""
    image = preprocess_image(str(img_path), size=tuple(config['IMAGE']['SIZE']))
    if image is None:
        return None
    
    features = extract_hog_features(image, config)
    
    label = None
    if not is_test:
        label = 0 if 'cat' in img_path.name.lower() else 1
        
    return features, label, img_path.name

def load_dataset(dataset_path: str, config: dict, max_samples: int = None, is_test: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load dataset and extract features with optional parallel processing."""
    if max_samples is None:
        max_samples = config['IMAGE']['MAX_SAMPLES']
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        logger.error(f"Dataset path {dataset_path} does not exist.")
        return np.array([]), None, []

    all_images = sorted(list(dataset_dir.glob('*.jpg')))
    
    # Shuffle images to get a balanced subset of classes
    import random
    random.seed(config['IMAGE'].get('RANDOM_SEED', 42))
    random.shuffle(all_images)
    
    if max_samples > 0:
        all_images = all_images[:max_samples]

    X, y, filenames = [], [], []

    logger.info(f"Loading and processing {len(all_images)} images from {dataset_path}...")
    
    # Using ProcessPoolExecutor for parallel feature extraction
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, all_images, [config]*len(all_images), [is_test]*len(all_images)), total=len(all_images)))

    for result in results:
        if result is not None:
            features, label, filename = result
            X.append(features)
            if label is not None:
                y.append(label)
            filenames.append(filename)

    X_arr = np.array(X)
    y_arr = np.array(y) if not is_test else None
    
    logger.info(f"Loaded {len(filenames)} images.")
    if not is_test:
        cat_count = np.sum(y_arr == 0)
        dog_count = np.sum(y_arr == 1)
        logger.info(f"Classes - Cat (0): {cat_count}, Dog (1): {dog_count}")
        
    return X_arr, y_arr, filenames
