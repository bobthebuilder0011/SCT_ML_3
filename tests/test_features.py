import unittest
import numpy as np
import cv2
import os
from src.features import preprocess_image, extract_hog_features
from src.utils import load_config

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create a dummy image for testing
        self.test_img_path = 'dummy_test.jpg'
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(self.test_img_path, dummy_img)
        self.config = load_config('config/config.yaml')

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)

    def test_preprocess_image(self):
        processed = preprocess_image(self.test_img_path, size=(64, 64))
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (64, 64))
        self.assertEqual(len(processed.shape), 2)  # Grayscale

    def test_extract_hog_features(self):
        processed = preprocess_image(self.test_img_path, size=(64, 64))
        features = extract_hog_features(processed, self.config)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)

if __name__ == '__main__':
    unittest.main()
