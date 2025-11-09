"""
Prediction Pipeline for Inference
Handles loading models and making predictions on new data.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from src.logger import logger
from src.exception import CustomException
from src.utils.common import load_object


class PredictionPipeline:
    """
    Pipeline for making predictions using trained models.
    Handles data preprocessing and model inference.
    """
    
    def __init__(
        self,
        model_path: str = "artifacts/models/xgboost.pkl",
        preprocessor_path: str = "artifacts/preprocessors/preprocessor.pkl",
        label_encoder_path: str = "artifacts/preprocessors/preprocessor_label_encoder.pkl"
    ):
        """
        Initialize prediction pipeline.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
            label_encoder_path: Path to label encoder
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.label_encoder_path = label_encoder_path
        
        # Load model and preprocessor
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model, preprocessor, and label encoder."""
        try:
            logger.info("Loading model and preprocessor for prediction")
            
            # Load model
            self.model = load_object(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
            
            # Load preprocessor
            self.preprocessor = load_object(self.preprocessor_path)
            logger.info(f"Preprocessor loaded from: {self.preprocessor_path}")
            
            # Load label encoder
            self.label_encoder = load_object(self.label_encoder_path)
            logger.info(f"Label encoder loaded from: {self.label_encoder_path}")
            
            logger.info("All artifacts loaded successfully")
            
        except Exception as e:
            logger.error("Error loading artifacts")
            raise CustomException(e, sys)
    
    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            input_data: Dictionary with customer features
            
        Returns:
            DataFrame with features ready for prediction
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Ensure all required columns are present
            required_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
            ]
            
            # Check for missing columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert SeniorCitizen to string for consistency with training
            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
            
            # Convert TotalCharges to numeric
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            logger.info(f"Features prepared. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error("Error preparing features")
            raise CustomException(e, sys)
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            input_data: Dictionary with customer features
            
        Returns:
            Dictionary with prediction results:
                - prediction: 'Yes' or 'No'
                - prediction_label: 1 or 0
                - probability: Probability of churn
                - confidence: Confidence score
        """
        try:
            logger.info("Starting prediction")
            
            # Prepare features
            df = self.prepare_features(input_data)
            
            # Transform features
            features_transformed = self.preprocessor.transform(df)
            logger.info(f"Features transformed. Shape: {features_transformed.shape}")
            
            # Make prediction
            prediction_encoded = self.model.predict(features_transformed)[0]
            prediction_proba = self.model.predict_proba(features_transformed)[0]
            
            # Decode prediction
            prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get probability of churn (class 1)
            churn_probability = float(prediction_proba[1])
            confidence = float(max(prediction_proba))
            
            result = {
                'prediction': prediction_label,
                'prediction_label': int(prediction_encoded),
                'churn_probability': round(churn_probability, 4),
                'no_churn_probability': round(float(prediction_proba[0]), 4),
                'confidence': round(confidence, 4),
                'risk_level': self._get_risk_level(churn_probability)
            }
            
            logger.info(f"Prediction completed: {result['prediction']}")
            return result
            
        except Exception as e:
            logger.error("Error during prediction")
            raise CustomException(e, sys)
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple customers.
        
        Args:
            input_data_list: List of dictionaries with customer features
            
        Returns:
            List of prediction results
        """
        try:
            logger.info(f"Starting batch prediction for {len(input_data_list)} customers")
            
            results = []
            for i, input_data in enumerate(input_data_list):
                try:
                    result = self.predict(input_data)
                    result['customer_index'] = i
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error predicting for customer {i}: {str(e)}")
                    results.append({
                        'customer_index': i,
                        'error': str(e),
                        'prediction': None
                    })
            
            logger.info(f"Batch prediction completed for {len(results)} customers")
            return results
            
        except Exception as e:
            logger.error("Error during batch prediction")
            raise CustomException(e, sys)
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a DataFrame of customers.
        
        Args:
            df: DataFrame with customer features
            
        Returns:
            DataFrame with predictions added
        """
        try:
            logger.info(f"Starting DataFrame prediction for {len(df)} customers")
            
            # Transform features
            features_transformed = self.preprocessor.transform(df)
            
            # Make predictions
            predictions_encoded = self.model.predict(features_transformed)
            predictions_proba = self.model.predict_proba(features_transformed)
            
            # Decode predictions
            predictions_labels = self.label_encoder.inverse_transform(predictions_encoded)
            
            # Add predictions to DataFrame
            df_results = df.copy()
            df_results['prediction'] = predictions_labels
            df_results['prediction_label'] = predictions_encoded
            df_results['churn_probability'] = predictions_proba[:, 1]
            df_results['confidence'] = predictions_proba.max(axis=1)
            df_results['risk_level'] = df_results['churn_probability'].apply(self._get_risk_level)
            
            logger.info("DataFrame prediction completed")
            return df_results
            
        except Exception as e:
            logger.error("Error during DataFrame prediction")
            raise CustomException(e, sys)
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Determine risk level based on churn probability.
        
        Args:
            probability: Churn probability
            
        Returns:
            Risk level: 'Low', 'Medium', 'High', or 'Critical'
        """
        if probability < 0.3:
            return 'Low'
        elif probability < 0.5:
            return 'Medium'
        elif probability < 0.7:
            return 'High'
        else:
            return 'Critical'
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model (if available).
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Get feature names from preprocessor
                feature_names = self.preprocessor.get_feature_names_out()
                importance_scores = self.model.feature_importances_
                
                importance_dict = dict(zip(feature_names, importance_scores))
                
                # Sort by importance
                importance_dict = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
                
                return importance_dict
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return {}
                
        except Exception as e:
            logger.error("Error getting feature importance")
            return {}


class CustomData:
    """
    Helper class to create input data for predictions.
    Makes it easier to create prediction requests.
    """
    
    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float
    ):
        """Initialize customer data."""
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prediction."""
        return {
            'gender': self.gender,
            'SeniorCitizen': self.SeniorCitizen,
            'Partner': self.Partner,
            'Dependents': self.Dependents,
            'tenure': self.tenure,
            'PhoneService': self.PhoneService,
            'MultipleLines': self.MultipleLines,
            'InternetService': self.InternetService,
            'OnlineSecurity': self.OnlineSecurity,
            'OnlineBackup': self.OnlineBackup,
            'DeviceProtection': self.DeviceProtection,
            'TechSupport': self.TechSupport,
            'StreamingTV': self.StreamingTV,
            'StreamingMovies': self.StreamingMovies,
            'Contract': self.Contract,
            'PaperlessBilling': self.PaperlessBilling,
            'PaymentMethod': self.PaymentMethod,
            'MonthlyCharges': self.MonthlyCharges,
            'TotalCharges': self.TotalCharges
        }