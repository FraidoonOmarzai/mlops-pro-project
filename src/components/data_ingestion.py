import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logger
from src.exception import CustomException
from pathlib import Path
from src.constants import CONFIG_PATH
from src.utils.common import read_yaml


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    """
    Handles data loading and splitting into train/test sets.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion component.

        Args:
            config: DataIngestionConfig object with paths and parameters
        """
        self.config = config
        logger.info("Data Ingestion component initialized")

    def initiate_data_ingestion(self) -> tuple:
        """
        Load data and split into train/test sets.

        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logger.info("Starting data ingestion process")

        try:
            # Read the dataset
            logger.info(f"Reading dataset from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Basic info logging
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Missing values: {df.isnull().sum().sum()}")
            logger.info(f"Duplicates: {df.duplicated().sum()}")

            # Create directory for processed data
            os.makedirs(os.path.dirname(
                self.config.train_data_path), exist_ok=True)

            # Split the data
            logger.info(
                f"Splitting data with test_size={self.config.test_size}")
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                # Stratify on target: while running test_data_ingestion.py, this line cause error 
                # stratify=df.iloc[:, -1] if 'Churn' in df.columns else None
            )

            logger.info(f"Train set shape: {train_set.shape}")
            logger.info(f"Test set shape: {test_set.shape}")

            # Save train and test sets
            train_set.to_csv(self.config.train_data_path,
                             index=False, header=True)
            test_set.to_csv(self.config.test_data_path,
                            index=False, header=True)

            logger.info("Data ingestion completed successfully")
            logger.info(f"Train data saved to: {self.config.train_data_path}")
            logger.info(f"Test data saved to: {self.config.test_data_path}")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error("Error in data ingestion")
            raise CustomException(e, sys)

    def get_data_info(self) -> dict:
        """
        Get information about the ingested data.

        Returns:
            Dictionary with data statistics
        """
        try:
            df = pd.read_csv(self.config.raw_data_path)

            info = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': int(df.duplicated().sum()),
                'dtypes': df.dtypes.astype(str).to_dict()
            }

            return info

        except Exception as e:
            raise CustomException(e, sys)


def create_data_ingestion_config(config_dict: dict) -> DataIngestionConfig:
    """
    Create DataIngestionConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DataIngestionConfig object
    """
    return DataIngestionConfig(
        raw_data_path=config_dict.raw_data_path,
        train_data_path=config_dict.train_data_path,
        test_data_path=config_dict.test_data_path,
        test_size=config_dict.test_size,
        random_state=config_dict.random_state)
