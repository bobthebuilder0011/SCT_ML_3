# Advanced Dogs vs. Cats Image Classifier with SVM

An advanced, production-ready machine learning pipeline for image classification using SVM with HOG features. This version is modular, highly configurable, and includes robust error handling, hyperparameter tuning, and a CLI interface.

## 🚀 Key Enhancements

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, feature extraction, model management, and utilities.
- **Configurable Pipeline**: Entire pipeline behavior controlled via `config/config.yaml`.
- **Command Line Interface (CLI)**: Easy operation for training, evaluation, and batch prediction.
- **Hyperparameter Tuning**: Integrated `GridSearchCV` for optimizing SVM performance.
- **Parallel Processing**: Multi-threaded feature extraction for faster data preparation.
- **Production Ready**: Includes model serialization (joblib), logging, and robust exception handling.
- **Reproducible**: Docker support for consistent execution across environments.
- **Developer Friendly**: Type hints, docstrings, and unit tests included.

## 📁 Project Structure

```text
.
├── src/                # Source code
│   ├── data_loader.py  # Parallel dataset loading
│   ├── features.py     # HOG & image preprocessing
│   ├── model.py        # SVM model & tuning logic
│   ├── utils.py        # Helpers & logging setup
│   └── main.py         # CLI entry point
├── config/             # Configuration files
│   └── config.yaml
├── tests/              # Unit tests
├── notebooks/          # Exploratory notebooks
├── output/             # Saved models and predictions
├── Dockerfile          # Containerization
├── requirements.txt    # Dependencies
└── setup.py            # Package installation
```

## 🛠️ Installation

### Standard Setup
1. **Clone & Install**:
   ```bash
   git clone <repo-url>
   cd SCT_ML_3
   pip install -e .
   ```

2. **Dependencies**:
   `pip install -r requirements.txt`

### Docker Setup
```bash
docker build -t dog-cat-svm .
docker run -v $(pwd)/dataset:/app/dataset -v $(pwd)/output:/app/output dog-cat-svm
```

## 💻 Usage (CLI)

The project provides a unified CLI via `dog-cat-svm` (after installation) or directly through `python run.py`.

### 1. Training
Basic training on 1000 samples:
```bash
python run.py --mode train --samples 1000
```

Training with **Hyperparameter Tuning**:
```bash
python run.py --mode train --tune --samples 2000
```

### 2. Evaluation
Evaluate an existing model on the training dataset:
```bash
python run.py --mode evaluate --model_path output/svm_model.joblib
```

### 3. Prediction
Run batch prediction on the test folder:
```bash
python run.py --mode predict --model_path output/svm_model.joblib
```

## ⚙️ Configuration

Modify `config/config.yaml` to adjust:
- Image dimensions (default 64x64)
- HOG parameters (orientations, pixels per cell, etc.)
- SVM settings (kernel, C, gamma)
- Training/Test split ratio

## 🧪 Testing

Run unit tests using:
```bash
python -m unittest discover tests
```

## 📊 Performance

By extracting **Histogram of Oriented Gradients (HOG)** features, we capture the essential structural information of cats and dogs, allowing a classic **Support Vector Machine (SVM)** to achieve high accuracy without the overhead of deep learning.

- **Typical Accuracy**: 80-85%
- **Speed**: Features extraction is parallelized for maximum efficiency.

## 📄 License
MIT
