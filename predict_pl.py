import pandas as pd
import pickle
import logging
import numpy as np
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CarPricePredictor:
    def __init__(self):
        """Initialize predictor with trained pipeline"""
        try:
            logger.info("Loading model components...")
            
            # Check if all required files exist
            required_files = ["preprocessor.pkl", "selector.pkl", "model.pkl"]
            for file in required_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"{file} not found. Please run the training pipeline first.")
            
            with open("preprocessor.pkl", "rb") as f:
                self.preprocessor = pickle.load(f)
            with open("selector.pkl", "rb") as f:
                self.selector = pickle.load(f)
            with open("model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            # Load feature info if available
            self.feature_info = None
            if os.path.exists("feature_info.pkl"):
                with open("feature_info.pkl", "rb") as f:
                    self.feature_info = pickle.load(f)
            
            logger.info("Model components loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            raise
    
    def validate_input(self, input_dict):
        """Validate input data types and ranges"""
        required_fields = {
            'Location': str,
            'Kilometers_Driven': (int, float),
            'Fuel_Type': str,
            'Transmission': str,
            'Owner_Type': str,
            'Mileage': (int, float),
            'Engine': (int, float),
            'Power': (int, float),
            'Seats': (int, float),
            'Brand': str,
            'Car_Age': (int, float)
        }
        
        # Check for missing fields
        missing_fields = []
        for field in required_fields:
            if field not in input_dict:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Type validation
        for field, expected_types in required_fields.items():
            if field in input_dict and not isinstance(input_dict[field], expected_types):
                raise ValueError(f"Invalid type for {field}. Expected {expected_types}, got {type(input_dict[field])}")
        
        # Range validation
        validations = [
            ('Kilometers_Driven', lambda x: x >= 0, "cannot be negative"),
            ('Car_Age', lambda x: 0 <= x <= 30, "must be between 0 and 30"),
            ('Seats', lambda x: 2 <= x <= 10, "must be between 2 and 10"),
            ('Mileage', lambda x: x > 0, "must be positive"),
            ('Engine', lambda x: x > 0, "must be positive"),
            ('Power', lambda x: x > 0, "must be positive")
        ]
        
        for field, validator, message in validations:
            if field in input_dict and not validator(input_dict[field]):
                raise ValueError(f"{field} {message}")
        
        # Categorical value suggestions (if we have training data info)
        categorical_fields = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
        for field in categorical_fields:
            if field in input_dict:
                # Convert to string and strip whitespace
                input_dict[field] = str(input_dict[field]).strip()
    
    def make_prediction(self, input_dict):
        """Make price prediction from input dictionary"""
        try:
            logger.info("Validating input data...")
            self.validate_input(input_dict)
            
            logger.info("Preprocessing input data...")
            input_df = pd.DataFrame([input_dict])
            
            # Transform through the pipeline
            input_processed = self.preprocessor.transform(input_df)
            input_selected = self.selector.transform(input_processed)
            
            logger.info("Making prediction...")
            prediction = self.model.predict(input_selected)
            
            # Ensure prediction is positive (prices can't be negative)
            prediction = max(0, prediction[0])
            
            logger.info(f"Prediction completed: {prediction:.2f}")
            
            # Return as float for JSON serialization
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_prediction_confidence(self, input_dict):
        """Get prediction with confidence intervals (for RandomForest)"""
        try:
            # Validate and preprocess
            self.validate_input(input_dict)
            input_df = pd.DataFrame([input_dict])
            input_processed = self.preprocessor.transform(input_df)
            input_selected = self.selector.transform(input_processed)
            
            # Get predictions from all trees
            if hasattr(self.model, 'estimators_'):
                tree_predictions = [tree.predict(input_selected)[0] for tree in self.model.estimators_]
                
                prediction = np.mean(tree_predictions)
                std = np.std(tree_predictions)
                
                # 95% confidence interval
                confidence_interval = (
                    max(0, prediction - 1.96 * std),
                    prediction + 1.96 * std
                )
                
                return {
                    'prediction': float(prediction),
                    'std': float(std),
                    'confidence_interval': confidence_interval
                }
            else:
                # Fallback for models without estimators
                prediction = self.make_prediction(input_dict)
                return {'prediction': prediction}
                
        except Exception as e:
            logger.error(f"Confidence prediction failed: {str(e)}")
            raise

# Singleton predictor instance
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = CarPricePredictor()
    return _predictor

def make_prediction(input_dict):
    """Wrapper function for making predictions"""
    predictor = get_predictor()
    return predictor.make_prediction(input_dict)

def make_prediction_with_confidence(input_dict):
    """Wrapper function for predictions with confidence"""
    predictor = get_predictor()
    return predictor.get_prediction_confidence(input_dict)