import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_data(df):
    """Clean and preprocess raw data"""
    try:
        logger.info(f"Initial data shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ["Unnamed: 0", "New_Price"]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df.drop(columns=existing_columns, inplace=True)
            logger.info(f"Dropped columns: {existing_columns}")
        
        # Extract numerical values from string columns with better error handling
        if "Mileage" in df.columns:
            df["Mileage"] = df["Mileage"].astype(str).str.extract(r"([\d.]+)").astype(float)
        
        if "Engine" in df.columns:
            df["Engine"] = df["Engine"].astype(str).str.extract(r"([\d.]+)").astype(float)
        
        if "Power" in df.columns:
            df["Power"] = (df["Power"].astype(str)
                          .str.replace('null', 'NaN', case=False)
                          .str.extract(r"([\d.]+)")
                          .astype(float))
        
        if "Seats" in df.columns:
            df["Seats"] = pd.to_numeric(df["Seats"], errors="coerce")
        
        # Add derived features
        current_year = datetime.now().year
        if "Year" in df.columns:
            df["Car_Age"] = current_year - df["Year"]
            df.drop(columns=["Year"], inplace=True)
        
        if "Name" in df.columns:
            df["Brand"] = df["Name"].str.split().str[0]
            df.drop(columns=["Name"], inplace=True)
        
        # Remove outliers using IQR method for numerical columns
        outlier_columns = ["Price", "Kilometers_Driven"]
        initial_shape = df.shape[0]
        
        for col in outlier_columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df = df[~outliers_mask]
                
                logger.info(f"Removed {outliers_mask.sum()} outliers from {col}")
        
        logger.info(f"Data shape after cleaning: {df.shape}")
        logger.info(f"Removed {initial_shape - df.shape[0]} rows total")
        
        return df
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

def build_preprocessor(num_features, cat_features):
    """Build preprocessing pipeline"""
    try:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ])
        
        logger.info(f"Built preprocessor with {len(num_features)} numerical and {len(cat_features)} categorical features")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Error building preprocessor: {str(e)}")
        raise

def main():
    try:
        # Check if required files exist
        if not os.path.exists("train-data.csv"):
            raise FileNotFoundError("train-data.csv not found in current directory")
        
        # Load and clean data
        logger.info("Loading and cleaning data...")
        df = pd.read_csv("train-data.csv")
        
        df = clean_data(df)
        df.to_csv("cleaned_data.csv", index=False)
        logger.info("Cleaned data saved to cleaned_data.csv")
        
        # Separate features and target
        if "Price" not in df.columns:
            raise ValueError("Target column 'Price' not found in data")
        
        X = df.drop("Price", axis=1)
        y = df["Price"]
        
        # Define features based on what's available in the data
        available_columns = X.columns.tolist()
        
        num_features = [col for col in ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "Car_Age"] 
                       if col in available_columns]
        cat_features = [col for col in ["Location", "Fuel_Type", "Transmission", "Owner_Type", "Brand"] 
                       if col in available_columns]
        
        logger.info(f"Numerical features: {num_features}")
        logger.info(f"Categorical features: {cat_features}")
        
        if not num_features and not cat_features:
            raise ValueError("No valid features found for preprocessing")
        
        # Build and fit preprocessor
        logger.info("Building preprocessing pipeline...")
        preprocessor = build_preprocessor(num_features, cat_features)
        X_processed = preprocessor.fit_transform(X)
        
        # Feature selection
        logger.info("Performing feature selection...")
        selector = SelectKBest(score_func=mutual_info_regression, k='all')
        selector.fit(X_processed, y)
        
        # Log feature scores
        feature_scores = selector.scores_
        logger.info(f"Feature selection completed. Max score: {np.max(feature_scores):.3f}")
        
        # Save pipeline components
        logger.info("Saving pipeline components...")
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
        with open("selector.pkl", "wb") as f:
            pickle.dump(selector, f)
        
        # Save feature names for reference
        feature_info = {
            'num_features': num_features,
            'cat_features': cat_features,
            'feature_scores': feature_scores.tolist()
        }
        with open("feature_info.pkl", "wb") as f:
            pickle.dump(feature_info, f)
        
        logger.info("Pipeline successfully built and saved")
        logger.info(f"Processed data shape: {X_processed.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()