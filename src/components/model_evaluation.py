
import os
import sys
import numpy as np
import pandas as pd
import mlflow
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from src.logger import logger
from src.exception import CustomException
from src.utils.common import save_json, read_yaml
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import CONFIG_PATH


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component."""
    metrics_path: str
    threshold: float = 0.5
    min_f1_score: float = 0.75
    min_roc_auc: float = 0.85


class ModelEvaluation:
    """
    Evaluates trained models and generates comprehensive metrics.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize ModelEvaluation component.

        Args:
            config: ModelEvaluationConfig object
        """
        self.config = config
        logger.info("Model Evaluation component initialized")

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}

            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(
                precision_score(y_true, y_pred, average='binary'))
            metrics['recall'] = float(recall_score(
                y_true, y_pred, average='binary'))
            metrics['f1_score'] = float(
                f1_score(y_true, y_pred, average='binary'))

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }

            # Derived metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = float(
                tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = float(
                tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            # ROC AUC if probabilities are available
            if y_pred_proba is not None:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))

            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = report

            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on train and test data.

        Args:
            model_name: Name of the model
            model: Trained model instance
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"Evaluating {model_name}...")

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Prediction probabilities
            y_train_pred_proba = None
            y_test_pred_proba = None

            if hasattr(model, 'predict_proba'):
                y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            train_metrics = self.calculate_metrics(
                y_train, y_train_pred, y_train_pred_proba)
            test_metrics = self.calculate_metrics(
                y_test, y_test_pred, y_test_pred_proba)

            evaluation_results = {
                'model_name': model_name,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'threshold': self.config.threshold
            }

            # Check if model meets minimum requirements
            meets_requirements = (
                test_metrics['f1_score'] >= self.config.min_f1_score and
                test_metrics.get('roc_auc', 0) >= self.config.min_roc_auc
            )

            evaluation_results['meets_requirements'] = meets_requirements

            # Log summary
            logger.info(f"\n{model_name} Evaluation Results:")
            logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
            logger.info(f"  Test Precision: {test_metrics['precision']:.4f}")
            logger.info(f"  Test Recall:    {test_metrics['recall']:.4f}")
            logger.info(f"  Test F1-Score:  {test_metrics['f1_score']:.4f}")
            if 'roc_auc' in test_metrics:
                logger.info(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
            logger.info(f"  Meets Requirements: {meets_requirements}")

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating {model_name}")
            raise CustomException(e, sys)

    def compare_models(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare all evaluated models and select the best one.

        Args:
            evaluation_results: Dictionary of evaluation results for all models

        Returns:
            Dictionary with comparison results and best model info
        """
        try:
            logger.info("\nComparing models...")

            comparison = []

            for model_name, results in evaluation_results.items():
                test_metrics = results['test_metrics']

                comparison.append({
                    'model_name': model_name,
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'f1_score': test_metrics['f1_score'],
                    'roc_auc': test_metrics.get('roc_auc', 0),
                    'meets_requirements': results['meets_requirements']
                })

            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values(
                'f1_score', ascending=False)

            # Select best model based on F1 score
            best_model = comparison_df.iloc[0].to_dict()

            logger.info("\nModel Comparison (sorted by F1-Score):")
            logger.info("\n" + comparison_df.to_string(index=False))
            logger.info(f"\nBest Model: {best_model['model_name']}")
            logger.info(f"  F1-Score: {best_model['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:  {best_model['roc_auc']:.4f}")

            return {
                'comparison_table': comparison_df.to_dict('records'),
                'best_model': best_model
            }

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(
        self,
        trained_models: Dict[str, Any],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate all trained models and compare them.

        Args:
            trained_models: Dictionary of trained models
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting model evaluation process")

        try:
            evaluation_results = {}

            # Evaluate each model
            for model_name, model_info in trained_models.items():
                model = model_info['model']

                results = self.evaluate_model(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test
                )

                evaluation_results[model_name] = results

            # Compare models
            comparison_results = self.compare_models(evaluation_results)

            # Compile final report
            final_report = {
                'individual_evaluations': evaluation_results,
                'comparison': comparison_results,
                'best_model_name': comparison_results['best_model']['model_name']
            }

            # Save evaluation report
            os.makedirs(self.config.metrics_path, exist_ok=True)
            report_path = os.path.join(
                self.config.metrics_path, 'evaluation_report.json')
            save_json(report_path, final_report)

            logger.info(f"\nEvaluation report saved to: {report_path}")
            logger.info("Model evaluation completed successfully")

            return final_report

        except Exception as e:
            logger.error("Error in model evaluation")
            raise CustomException(e, sys)


def create_model_evaluation_config(config_dict: dict) -> ModelEvaluationConfig:
    """
    Create ModelEvaluationConfig from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        ModelEvaluationConfig object
    """
    return ModelEvaluationConfig(
        metrics_path=config_dict.metrics_path,
        threshold=config_dict.threshold,
        min_f1_score=config_dict.min_f1_score,
        min_roc_auc=config_dict.min_roc_auc
    )
