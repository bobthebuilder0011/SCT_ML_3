import logging
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SVMModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        self.svm_config = config['SVM']

    def preprocess_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standardize features."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def train(self, X: np.ndarray, y: np.ndarray, tune: bool = False):
        """Train the SVM model with optional hyperparameter tuning."""
        X_scaled = self.preprocess_features(X, fit=True)
        
        if tune:
            logger.info("Starting hyperparameter tuning with GridSearchCV...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'linear']
            }
            grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=1, cv=3)
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_
            logger.info(f"Best parameters: {grid.best_params_}")
        else:
            logger.info("Training SVM with provided config...")
            self.model = SVC(
                kernel=self.svm_config['KERNEL'],
                C=self.svm_config['C'],
                gamma=self.svm_config['GAMMA'],
                probability=self.svm_config['PROBABILITY'],
                random_state=42
            )
            self.model.fit(X_scaled, y)
        
        logger.info("Training complete.")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        X_scaled = self.preprocess_features(X)
        y_pred = self.model.predict(X_scaled)
        
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=['Cat', 'Dog'], output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict classes and probabilities for new data."""
        X_scaled = self.preprocess_features(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled) if self.svm_config['PROBABILITY'] else None
        return predictions, probabilities

    def save_model(self, path: str):
        """Save the model and scaler to disk."""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model and scaler from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        logger.info(f"Model loaded from {path}")
