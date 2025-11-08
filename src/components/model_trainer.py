import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.logger import logger
from src.exception import CustomException
from src.utils.common import save_object, read_yaml
from src.constants import CONFIG_PATH, MODEL_CONFIG_PATH


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer component."""
    models_dir: str
    models: list
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class ModelTrainer:
    """
    Trains multiple ML models and tracks experiments with MLflow.
    """

    def __init__(self, config: ModelTrainerConfig, model_params: Dict[str, Dict]):
        """
        Initialize ModelTrainer component.

        Args:
            config: ModelTrainerConfig object
            model_params: Dictionary of model hyperparameters
        """
        self.config = config
        self.model_params = model_params
        self.models = {}
        self.trained_models = {}

        # Setup MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

        logger.info("Model Trainer component initialized")
        logger.info(f"MLflow tracking URI: {config.mlflow_tracking_uri}")
        logger.info(f"MLflow experiment: {config.mlflow_experiment_name}")

    def get_models(self) -> Dict[str, Any]:
        """
        Initialize models with their hyperparameters.

        Returns:
            Dictionary of model instances
        """
        try:
            models = {}

            if 'logistic_regression' in self.config.models:
                models['logistic_regression'] = LogisticRegression(
                    **self.model_params.get('logistic_regression', {})
                )

            if 'random_forest' in self.config.models:
                models['random_forest'] = RandomForestClassifier(
                    **self.model_params.get('random_forest', {})
                )

            if 'xgboost' in self.config.models:
                models['xgboost'] = XGBClassifier(
                    **self.model_params.get('xgboost', {})
                )

            if 'lightgbm' in self.config.models:
                models['lightgbm'] = LGBMClassifier(
                    **self.model_params.get('lightgbm', {}),
                    verbose=-1
                )

            logger.info(
                f"Initialized {len(models)} models: {list(models.keys())}")
            return models

        except Exception as e:
            raise CustomException(e, sys)

    def train_model(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, str]:
        """
        Train a single model and log to MLflow.

        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Tuple of (trained_model, model_path)
        """
        try:
            logger.info(f"Training {model_name}...")

            with mlflow.start_run(run_name=f"{model_name}_run") as run:
                # Log model parameters
                mlflow.log_params(self.model_params.get(model_name, {}))

                # Train model
                model.fit(X_train, y_train)
                logger.info(f"{model_name} training completed")

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_train_pred_proba = None
                    y_test_pred_proba = None

                # Calculate basic metrics (detailed evaluation in model_evaluation.py)
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(
                    y_test, y_test_pred, average='binary')
                test_recall = recall_score(
                    y_test, y_test_pred, average='binary')
                test_f1 = f1_score(y_test, y_test_pred, average='binary')

                # Log metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1_score", test_f1)

                logger.info(
                    f"{model_name} - Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

                # Log model to MLflow
                mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Save model locally
                model_path = os.path.join(
                    self.config.models_dir, f"{model_name}.pkl")
                save_object(model_path, model)
                logger.info(f"{model_name} saved to: {model_path}")

                # Log artifact path
                mlflow.log_param("model_path", model_path)

                return model, model_path

        except Exception as e:
            logger.error(f"Error training {model_name}")
            raise CustomException(e, sys)

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all configured models.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Dictionary with trained models and their paths
        """
        logger.info("Starting model training process")

        try:
            # Get models
            self.models = self.get_models()

            logger.info(f"Training {len(self.models)} models")
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Test data shape: {X_test.shape}")

            # Train each model
            results = {}

            for model_name, model in self.models.items():
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {model_name}")
                logger.info(f"{'='*50}")

                trained_model, model_path = self.train_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )

                results[model_name] = {
                    'model': trained_model,
                    'model_path': model_path
                }

                self.trained_models[model_name] = trained_model

            logger.info(f"\n{'='*50}")
            logger.info("Model training completed for all models")
            logger.info(f"{'='*50}")

            return results

        except Exception as e:
            logger.error("Error in model training")
            raise CustomException(e, sys)


def create_model_trainer_config(config_dict: dict, mlflow_config: dict) -> ModelTrainerConfig:
    """
    Create ModelTrainerConfig from dictionaries.

    Args:
        config_dict: Model training configuration dictionary
        mlflow_config: MLflow configuration dictionary

    Returns:
        ModelTrainerConfig object
    """
    return ModelTrainerConfig(
        models_dir=config_dict.models_dir,
        models=config_dict.models,
        mlflow_tracking_uri=mlflow_config.tracking_uri,
        mlflow_experiment_name=mlflow_config.experiment_name
    )
