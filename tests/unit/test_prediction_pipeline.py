import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

@pytest.fixture
def prediction_pipeline_mock():
    """
    Returns a PredictionPipeline with model, preprocessor, and label encoder mocked.
    Allows testing the pipeline logic without real artifacts.
    """
    with patch("src.pipeline.prediction_pipeline.load_object") as mock_load:
        # Mock model
        fake_model = MagicMock()
        fake_model.predict.return_value = [1]                  # encoded prediction
        fake_model.predict_proba.return_value = [[0.3, 0.7]]  # [no_churn, churn]
        
        # Mock preprocessor
        fake_preprocessor = MagicMock()
        fake_preprocessor.transform.return_value = np.array([[0.1]*10])  # arbitrary transformed features
        
        # Mock label encoder
        fake_label_encoder = MagicMock()
        fake_label_encoder.inverse_transform.return_value = ['Yes']
        
        # Set mock_load return order: model, preprocessor, label encoder
        mock_load.side_effect = [fake_model, fake_preprocessor, fake_label_encoder]
        
        pipeline = PredictionPipeline()
        yield pipeline


@pytest.mark.unit
@pytest.mark.requires_model
class TestPredictionPipeline:
    
    def test_prediction_pipeline_initialization(self, prediction_pipeline_mock):
        assert prediction_pipeline_mock.model is not None
        assert prediction_pipeline_mock.preprocessor is not None
        assert prediction_pipeline_mock.label_encoder is not None
    
    def test_prepare_features(self, prediction_pipeline_mock):
        sample_customer_data = CustomData(
            gender='Male',
            SeniorCitizen=0,
            Partner='Yes',
            Dependents='No',
            tenure=12,
            PhoneService='Yes',
            MultipleLines='No',
            InternetService='DSL',
            OnlineSecurity='No',
            OnlineBackup='Yes',
            DeviceProtection='No',
            TechSupport='No',
            StreamingTV='No',
            StreamingMovies='No',
            Contract='Month-to-month',
            PaperlessBilling='Yes',
            PaymentMethod='Electronic check',
            MonthlyCharges=50.0,
            TotalCharges=600.0
        ).to_dict()
        
        df = prediction_pipeline_mock.prepare_features(sample_customer_data)
        assert isinstance(df, pd.DataFrame)
        assert 'gender' in df.columns
        assert 'tenure' in df.columns
        assert df['SeniorCitizen'].dtype == object  # converted to string
    
    def test_predict_single_customer(self, prediction_pipeline_mock):
        sample_customer_data = CustomData(
            gender='Male',
            SeniorCitizen=0,
            Partner='Yes',
            Dependents='No',
            tenure=12,
            PhoneService='Yes',
            MultipleLines='No',
            InternetService='DSL',
            OnlineSecurity='No',
            OnlineBackup='Yes',
            DeviceProtection='No',
            TechSupport='No',
            StreamingTV='No',
            StreamingMovies='No',
            Contract='Month-to-month',
            PaperlessBilling='Yes',
            PaymentMethod='Electronic check',
            MonthlyCharges=50.0,
            TotalCharges=600.0
        ).to_dict()
        
        result = prediction_pipeline_mock.predict(sample_customer_data)
        assert result['prediction'] == 'Yes'
        assert result['prediction_label'] == 1
        assert result['churn_probability'] == 0.7
        assert result['no_churn_probability'] == 0.3
        assert result['confidence'] == 0.7
        assert result['risk_level'] == 'Critical'
    
    def test_risk_level_calculation(self):
        assert PredictionPipeline._get_risk_level(0.2) == 'Low'
        assert PredictionPipeline._get_risk_level(0.4) == 'Medium'
        assert PredictionPipeline._get_risk_level(0.6) == 'High'
        assert PredictionPipeline._get_risk_level(0.8) == 'Critical'
    
    def test_custom_data_class(self):
        customer = CustomData(
            gender='Female',
            SeniorCitizen=1,
            Partner='No',
            Dependents='No',
            tenure=24,
            PhoneService='Yes',
            MultipleLines='Yes',
            InternetService='Fiber optic',
            OnlineSecurity='Yes',
            OnlineBackup='No',
            DeviceProtection='Yes',
            TechSupport='No',
            StreamingTV='Yes',
            StreamingMovies='No',
            Contract='Two year',
            PaperlessBilling='No',
            PaymentMethod='Mailed check',
            MonthlyCharges=80.0,
            TotalCharges=1920.0
        )
        data_dict = customer.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['gender'] == 'Female'
        assert data_dict['tenure'] == 24
        assert data_dict['MonthlyCharges'] == 80.0