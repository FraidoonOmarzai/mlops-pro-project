import os
import sys
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from src.logger import logger
from src.exception import CustomException
from src.utils.common import read_yaml,  save_json
from src.constants import CONFIG_PATH


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    report_path: str


class DataValidation:
    """
    Validates data quality and schema compliance.
    """

    # Expected schema for churn dataset
    EXPECTED_COLUMNS = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]

    # type of TotalCharges is object
    # NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges']

    CATEGORICAL_COLUMNS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    TARGET_COLUMN = 'Churn'

    def __init__(self, config: DataValidationConfig):
        """
        Initialize DataValidation component.

        Args:
            config: DataValidationConfig object
        """
        self.config = config
        logger.info("Data Validation component initialized")

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate if dataframe matches expected schema.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {}

            # Check if all expected columns are present
            missing_columns = set(self.EXPECTED_COLUMNS) - set(df.columns)
            validation_results['all_columns_present'] = len(
                missing_columns) == 0
            validation_results['missing_columns'] = list(missing_columns)

            # Check for extra columns
            extra_columns = set(df.columns) - set(self.EXPECTED_COLUMNS)
            validation_results['extra_columns'] = list(extra_columns)

            # Check data types for numerical columns
            numerical_dtype_check = {}
            for col in self.NUMERICAL_COLUMNS:
                if col in df.columns:
                    numerical_dtype_check[col] = pd.api.types.is_numeric_dtype(
                        df[col])
            validation_results['numerical_dtypes_correct'] = all(
                numerical_dtype_check.values())
            validation_results['numerical_dtype_details'] = numerical_dtype_check

            logger.info(f"Schema validation completed: {validation_results}")
            return validation_results

        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with quality check results
        """
        try:
            quality_report = {}

            # Check for missing values
            missing_values = df.isnull().sum()
            quality_report['missing_values'] = missing_values[missing_values > 0].to_dict(
            )
            quality_report['total_missing'] = int(df.isnull().sum().sum())
            quality_report['missing_percentage'] = round(
                (df.isnull().sum().sum() /
                 (df.shape[0] * df.shape[1])) * 100, 2
            )

            # Check for duplicates
            quality_report['duplicate_rows'] = int(df.duplicated().sum())
            quality_report['duplicate_percentage'] = round(
                (df.duplicated().sum() / len(df)) * 100, 2
            )

            # Check for data ranges (numerical columns)
            numerical_stats = {}
            for col in self.NUMERICAL_COLUMNS:
                if col in df.columns:
                    numerical_stats[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'negative_values': int((df[col] < 0).sum())
                    }
            quality_report['numerical_statistics'] = numerical_stats

            # Check target distribution
            if self.TARGET_COLUMN in df.columns:
                target_dist = df[self.TARGET_COLUMN].value_counts()
                quality_report['target_distribution'] = target_dist.to_dict()
                quality_report['target_balance_ratio'] = round(
                    target_dist.min() / target_dist.max(), 2
                )

            logger.info(f"Data quality validation completed")
            return quality_report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, train_path: str, test_path: str) -> bool:
        """
        Perform complete data validation on train and test sets.

        Args:
            train_path: Path to training data
            test_path: Path to test data

        Returns:
            Boolean indicating if data passed validation
        """
        logger.info("Starting data validation process")

        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Loaded train data: {train_df.shape}")
            logger.info(f"Loaded test data: {test_df.shape}")

            # Validate schema
            train_schema = self.validate_schema(train_df)
            test_schema = self.validate_schema(test_df)

            # Validate quality
            train_quality = self.validate_data_quality(train_df)
            test_quality = self.validate_data_quality(test_df)

            # Compile validation report
            validation_report = {
                'train_data': {
                    'shape': train_df.shape,
                    'schema_validation': train_schema,
                    'quality_validation': train_quality
                },
                'test_data': {
                    'shape': test_df.shape,
                    'schema_validation': test_schema,
                    'quality_validation': test_quality
                },
                'validation_passed': (
                    train_schema['all_columns_present'] and
                    test_schema['all_columns_present']
                )
            }

            # Save validation report
            save_json(self.config.report_path, validation_report)
            logger.info(
                f"Validation report saved to: {self.config.report_path}")

            # Log critical issues
            if not validation_report['validation_passed']:
                logger.warning(
                    "Data validation failed! Check the validation report.")
            else:
                logger.info("Data validation passed successfully!")

            return validation_report['validation_passed']

        except Exception as e:
            logger.error("Error in data validation")
            raise CustomException(e, sys)


def create_data_validation_config(config_dict: dict) -> DataValidationConfig:
    """
    Create DataValidationConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DataValidationConfig object
    """
    return DataValidationConfig(
        report_path=config_dict.report_path)
