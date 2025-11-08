import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logger
from src.exception import CustomException
from src.utils.common import save_object
from src.constants import CONFIG_PATH
from src.utils.common import read_yaml


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing component."""
    preprocessor_path: str
    numerical_features: list
    categorical_features: list
    target_column: str


class DataPreprocessing:
    """
    Handles feature engineering and data transformation.
    """

    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize DataPreprocessing component.

        Args:
            config: DataPreprocessingConfig object
        """
        self.config = config
        self.preprocessor = None
        logger.info("Data Preprocessing component initialized")

    def get_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numerical and categorical features.

        Returns:
            ColumnTransformer object with preprocessing pipelines
        """
        try:
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first',
                     sparse_output=False, handle_unknown='ignore'))
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, self.config.numerical_features),
                    ('cat', categorical_pipeline, self.config.categorical_features)
                ],
                remainder='drop'
            )

            logger.info("Preprocessing pipeline created successfully")
            logger.info(
                f"Numerical features: {self.config.numerical_features}")
            logger.info(
                f"Categorical features: {self.config.categorical_features}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data cleaning operations.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Starting data cleaning")
            df_clean = df.copy()

            # Remove customerID column (not useful for modeling)
            if 'customerID' in df_clean.columns:
                df_clean = df_clean.drop('customerID', axis=1)
                logger.info("Dropped customerID column")

            # Handle TotalCharges - convert to numeric
            if 'TotalCharges' in df_clean.columns:
                df_clean['TotalCharges'] = pd.to_numeric(
                    df_clean['TotalCharges'],
                    errors='coerce'
                )
                logger.info("Converted TotalCharges to numeric")

            # Convert SeniorCitizen to object for categorical encoding
            if 'SeniorCitizen' in df_clean.columns:
                df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(
                    str)

            # Remove duplicates
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_duplicates = initial_rows - len(df_clean)
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate rows")

            logger.info(
                f"Data cleaning completed. Final shape: {df_clean.shape}")
            return df_clean

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_preprocessing(self, train_path: str, test_path: str) -> tuple:
        """
        Perform complete data preprocessing.

        Args:
            train_path: Path to training data
            test_path: Path to test data

        Returns:
            Tuple of (train_features, test_features, train_target, test_target, preprocessor_path)
        """
        logger.info("Starting data preprocessing process")

        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Loaded train data: {train_df.shape}")
            logger.info(f"Loaded test data: {test_df.shape}")

            # Clean data
            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            # Separate features and target
            target_column = self.config.target_column

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.info(f"Separated features and target")
            logger.info(
                f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(
                f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Encode target variable
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            logger.info(
                f"Target encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

            # Get preprocessing pipeline
            self.preprocessor = self.get_preprocessor()

            # Fit and transform training data
            logger.info("Fitting preprocessor on training data")
            X_train_transformed = self.preprocessor.fit_transform(X_train)

            # Transform test data
            logger.info("Transforming test data")
            X_test_transformed = self.preprocessor.transform(X_test)

            logger.info(
                f"Transformed X_train shape: {X_train_transformed.shape}")
            logger.info(
                f"Transformed X_test shape: {X_test_transformed.shape}")

            # Save preprocessor
            save_object(self.config.preprocessor_path, self.preprocessor)
            logger.info(
                f"Preprocessor saved to: {self.config.preprocessor_path}")

            # Also save label encoder
            label_encoder_path = self.config.preprocessor_path.replace(
                '.pkl', '_label_encoder.pkl')
            save_object(label_encoder_path, label_encoder)
            logger.info(f"Label encoder saved to: {label_encoder_path}")

            logger.info("Data preprocessing completed successfully")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train_encoded,
                y_test_encoded,
                self.config.preprocessor_path
            )

        except Exception as e:
            logger.error("Error in data preprocessing")
            raise CustomException(e, sys)


def create_data_preprocessing_config(config_dict: dict) -> DataPreprocessingConfig:
    """
    Create DataPreprocessingConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DataPreprocessingConfig object
    """
    return DataPreprocessingConfig(
        preprocessor_path=config_dict.preprocessor_path,
        numerical_features=config_dict.numerical_features,
        categorical_features=config_dict.categorical_features,
        target_column=config_dict.target_column
    )
