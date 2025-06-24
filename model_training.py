import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare data"""
    try:
        if not os.path.exists("cleaned_data.csv"):
            raise FileNotFoundError("cleaned_data.csv not found. Please run data_cleaning_pipeline.py first")
        
        df = pd.read_csv("cleaned_data.csv")
        logger.info(f"Loaded data with shape: {df.shape}")
        
        if "Price" not in df.columns:
            raise ValueError("Target column 'Price' not found in cleaned data")
        
        X = df.drop("Price", axis=1)
        y = df["Price"]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_pipeline():
    """Load preprocessing pipeline"""
    try:
        required_files = ["preprocessor.pkl", "selector.pkl"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} not found. Please run data_cleaning_pipeline.py first")
        
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        with open("selector.pkl", "rb") as f:
            selector = pickle.load(f)
        
        logger.info("Pipeline components loaded successfully")
        return preprocessor, selector
        
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    try:
        # Cross-validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        # Test metrics
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log results
        logger.info("="*50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Cross-validation R2 scores: {cv_scores}")
        logger.info(f"Mean CV R2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        logger.info("-"*30)
        logger.info("TRAINING METRICS:")
        logger.info(f"  R2 Score: {train_r2:.4f}")
        logger.info(f"  RMSE: {train_rmse:.2f}")
        logger.info(f"  MAE: {train_mae:.2f}")
        logger.info("-"*30)
        logger.info("TEST METRICS:")
        logger.info(f"  R2 Score: {test_r2:.4f}")
        logger.info(f"  RMSE: {test_rmse:.2f}")
        logger.info(f"  MAE: {test_mae:.2f}")
        logger.info("="*50)
        
        # Check for overfitting
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            logger.warning(f"Potential overfitting detected. Training R2 - Test R2 = {r2_diff:.4f}")
        
        return {
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def train_model():
    """Train and save the model"""
    try:
        # Load data and pipeline
        logger.info("Loading data and pipeline...")
        X, y = load_data()
        preprocessor, selector = load_pipeline()
        
        # Transform data
        logger.info("Transforming data...")
        X_processed = preprocessor.transform(X)
        X_selected = selector.transform(X_processed)
        
        logger.info(f"Final feature shape: {X_selected.shape}")
        
        # Split data for proper evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train model with optimized hyperparameters
        logger.info("Training model...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            logger.info(f"Top 5 most important features by index: {np.argsort(feature_importance)[-5:][::-1]}")
        
        # Save model and metrics
        logger.info("Saving model and metrics...")
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("model_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)
        
        logger.info("Model training completed successfully")
        logger.info(f"Final model test R2: {metrics['test_r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
