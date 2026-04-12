import argparse
import os
import pandas as pd
import logging
from .utils import load_config, setup_logging, create_directory
from .data_loader import load_dataset
from .model import SVMModel
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description="Dogs vs. Cats Image Classifier with SVM")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], default='train', help='Operation mode')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning during training')
    parser.add_argument('--model_path', type=str, default='output/svm_model.joblib', help='Path to save/load the model')
    parser.add_argument('--samples', type=int, help='Max number of samples to load (overrides config)')
    parser.add_argument('--test_image', type=str, help='Single image path for prediction')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging()
    logger = logging.getLogger(__name__)
    
    create_directory('output')
    
    if args.samples:
        config['IMAGE']['MAX_SAMPLES'] = args.samples
        
    model = SVMModel(config)
    
    if args.mode == 'train':
        X, y, _ = load_dataset(config['TRAIN_PATH'], config)
        if len(X) == 0:
            logger.error("No training data found.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['TRAINING']['TEST_SIZE'], 
            random_state=config['IMAGE']['RANDOM_SEED'], 
            stratify=y
        )
        
        model.train(X_train, y_train, tune=args.tune)
        model.save_model(args.model_path)
        
        logger.info("Evaluating on hold-out set:")
        metrics = model.evaluate(X_test, y_test)
        
    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            logger.error(f"Model not found at {args.model_path}. Please train a model first.")
            return
            
        model.load_model(args.model_path)
        X, y, _ = load_dataset(config['TRAIN_PATH'], config)
        if len(X) == 0:
            logger.error("No data found for evaluation.")
            return
            
        metrics = model.evaluate(X, y)
        
    elif args.mode == 'predict':
        if not os.path.exists(args.model_path):
            logger.error(f"Model not found at {args.model_path}. Please train a model first.")
            return
            
        model.load_model(args.model_path)
        
        if args.test_image:
            # Single image prediction logic could be added here
            # For now, let's stick to batch prediction as per notebook
            pass
            
        X_test, _, filenames = load_dataset(config['TEST_PATH'], config, is_test=True)
        if len(X_test) == 0:
            logger.error("No test images found.")
            return
            
        predictions, probabilities = model.predict(X_test)
        
        results = []
        for name, pred, prob in zip(filenames, predictions, probabilities):
            results.append({
                'image': name,
                'prediction': 'Dog' if pred == 1 else 'Cat',
                'confidence': prob[pred] if prob is not None else 0
            })
            
        results_df = pd.DataFrame(results)
        output_csv = 'output/test_predictions.csv'
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
