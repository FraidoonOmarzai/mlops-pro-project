"""
Script to download Telco Customer Churn dataset.
You can download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

For this project, you have two options:
1. Download manually from Kaggle
2. Use Kaggle API (requires kaggle credentials)
"""

import os
import sys
import requests
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import logger
from src.exception import CustomException


def download_from_url():
    """
    Download dataset from a direct URL.
    Note: You might need to update this URL or use Kaggle API.
    """
    try:
        # Create data directory
        os.makedirs("data/raw", exist_ok=True)
        
        # Alternative: Download from a mirror or use sample data
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        
        logger.info(f"Downloading dataset from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the file
        output_path = "data/raw/churn_data.csv"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Dataset downloaded successfully to {output_path}")
        
        # Verify the download
        df = pd.read_csv(output_path)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise CustomException(e, sys)


def download_with_kaggle_api():
    """
    Download dataset using Kaggle API.
    Requires: pip install kaggle
    Requires: ~/.kaggle/kaggle.json with API credentials
    """
    try:
        import kaggle
        
        os.makedirs("data/raw", exist_ok=True)
        
        logger.info("Downloading dataset from Kaggle")
        
        # Download from Kaggle
        kaggle.api.dataset_download_files(
            'blastchar/telco-customer-churn',
            path='data/raw',
            unzip=True
        )
        
        # Rename file
        os.rename(
            'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
            'data/raw/churn_data.csv'
        )
        
        logger.info("Dataset downloaded successfully from Kaggle")
        
        return "data/raw/churn_data.csv"
        
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {str(e)}")
        logger.info("Make sure you have:")
        logger.info("1. Installed kaggle: pip install kaggle")
        logger.info("2. Set up credentials: ~/.kaggle/kaggle.json")
        raise CustomException(e, sys)


def create_sample_data():
    """
    Create a small sample dataset for testing (if you can't download).
    This is just for demonstration purposes.
    """
    try:
        logger.info("Creating sample dataset for testing")
        
        # Sample data structure
        sample_data = {
            'customerID': [f'ID{i}' for i in range(100)],
            'gender': ['Male', 'Female'] * 50,
            'SeniorCitizen': [0, 1] * 50,
            'Partner': ['Yes', 'No'] * 50,
            'Dependents': ['Yes', 'No'] * 50,
            'tenure': list(range(1, 101)),
            'PhoneService': ['Yes', 'No'] * 50,
            'MultipleLines': ['Yes', 'No', 'No phone service'] * 33 + ['Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No'] * 33 + ['DSL'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'TechSupport': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'StreamingTV': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'StreamingMovies': ['Yes', 'No', 'No internet service'] * 33 + ['Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year'] * 33 + ['Month-to-month'],
            'PaperlessBilling': ['Yes', 'No'] * 50,
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'] * 25,
            'MonthlyCharges': [round(20 + i * 0.5, 2) for i in range(100)],
            'TotalCharges': [round((20 + i * 0.5) * (i + 1), 2) for i in range(100)],
            'Churn': ['Yes', 'No'] * 50
        }
        
        df = pd.DataFrame(sample_data)
        
        os.makedirs("data/raw", exist_ok=True)
        output_path = "data/raw/churn_data.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Sample dataset created at {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        
        return output_path
        
    except Exception as e:
        raise CustomException(e, sys)


def main():
    """
    Main function to download dataset.
    """
    logger.info("Starting dataset download process")
    
    try:
        # Try methods in order
        methods = [
            ("Direct URL", download_from_url),
            ("Kaggle API", download_with_kaggle_api),
            ("Sample Data", create_sample_data)
        ]
        
        for method_name, method_func in methods:
            try:
                logger.info(f"\nAttempting to download using: {method_name}")
                output_path = method_func()
                logger.info(f"\n{'='*50}")
                logger.info("DATASET DOWNLOAD SUCCESSFUL!")
                logger.info(f"{'='*50}")
                logger.info(f"Dataset saved at: {output_path}")
                logger.info("You can now run the training pipeline!")
                return output_path
            except Exception as e:
                logger.warning(f"{method_name} failed: {str(e)}")
                continue
        
        logger.error("All download methods failed!")
        logger.info("\nPlease download the dataset manually from:")
        logger.info("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        logger.info("And save it to: data/raw/churn_data.csv")
        
    except Exception as e:
        logger.error("Error in download process")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()