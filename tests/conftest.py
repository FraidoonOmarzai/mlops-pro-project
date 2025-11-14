"""
Pytest fixtures and configuration for all tests.
Shared fixtures available to all test modules.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.constants import CONFIG_PATH, MODEL_CONFIG_PATH
from src.utils.common import read_yaml
# from src.configs import ConfigManager
from src.pipeline.prediction_pipeline import PredictionPipeline


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    return read_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


# ==================== DATA FIXTURES ====================

@pytest.fixture(scope="session")
def sample_customer_data():
    """Sample customer data for testing."""
    return {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 840.50
    }


@pytest.fixture(scope="session")
def sample_dataframe():
    """Sample DataFrame for testing."""
    data = {
        'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [12, 24, 6, 48, 36],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'No', 'No internet service', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'No internet service', 'Yes', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'TechSupport': ['No', 'No', 'No internet service', 'Yes', 'No'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check', 
                         'Bank transfer (automatic)', 'Credit card (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 19.90, 89.15, 104.80],
        'TotalCharges': [358.20, 1367.80, 119.40, 4278.20, 3769.80],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(sample_dataframe, temp_dir):
    """Create temporary CSV file for testing."""
    csv_path = os.path.join(temp_dir, "test_data.csv")
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


# ==================== MODEL FIXTURES ====================

@pytest.fixture(scope="session")
def mock_model():
    """Mock trained model for testing."""
    from sklearn.linear_model import LogisticRegression
    
    # Create simple mock model
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    return model


@pytest.fixture(scope="session")
def mock_preprocessor():
    """Mock preprocessor for testing."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, [0, 1, 2])
        ]
    )
    
    # Fit with dummy data
    X = np.random.rand(10, 3)
    preprocessor.fit(X)
    
    return preprocessor


@pytest.fixture
def prediction_pipeline_mock(mock_model, mock_preprocessor, temp_dir, monkeypatch):
    """Mock prediction pipeline for testing."""
    # Mock the load_object function to return our mocks
    def mock_load_object(path):
        if 'model' in path:
            return mock_model
        elif 'preprocessor' in path:
            return mock_preprocessor
        elif 'encoder' in path:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(['No', 'Yes'])
            return le
    
    monkeypatch.setattr('src.utils.common.load_object', mock_load_object)
    
    return PredictionPipeline(
        model_path=os.path.join(temp_dir, "model.pkl"),
        preprocessor_path=os.path.join(temp_dir, "preprocessor.pkl"),
        label_encoder_path=os.path.join(temp_dir, "encoder.pkl")
    )


# ==================== API FIXTURES ====================

@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    return TestClient(app)


# ==================== HELPER FIXTURES ====================

@pytest.fixture
def mock_logger(monkeypatch):
    """Mock logger to suppress output during tests."""
    import logging
    
    mock_log = logging.getLogger('test')
    mock_log.setLevel(logging.CRITICAL)
    
    monkeypatch.setattr('src.logger.logger', mock_log)
    return mock_log


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow to avoid actual tracking during tests."""
    class MockMLflow:
        @staticmethod
        def set_tracking_uri(uri):
            pass
        
        @staticmethod
        def set_experiment(name):
            pass
        
        @staticmethod
        def start_run(*args, **kwargs):
            class MockRun:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return MockRun()
        
        @staticmethod
        def log_param(*args, **kwargs):
            pass
        
        @staticmethod
        def log_metric(*args, **kwargs):
            pass
        
        @staticmethod
        def log_params(*args, **kwargs):
            pass
        
        @staticmethod
        def log_metrics(*args, **kwargs):
            pass
    
    monkeypatch.setattr('mlflow.set_tracking_uri', MockMLflow.set_tracking_uri)
    monkeypatch.setattr('mlflow.set_experiment', MockMLflow.set_experiment)
    monkeypatch.setattr('mlflow.start_run', MockMLflow.start_run)
    monkeypatch.setattr('mlflow.log_param', MockMLflow.log_param)
    monkeypatch.setattr('mlflow.log_metric', MockMLflow.log_metric)
    
    return MockMLflow


# ==================== PARAMETRIZE DATA ====================

@pytest.fixture
def valid_customer_inputs():
    """List of valid customer inputs for parametrized tests."""
    return [
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': 29.85
        },
        {
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 72,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 118.75,
            'TotalCharges': 8550.00
        }
    ]


@pytest.fixture
def invalid_customer_inputs():
    """List of invalid customer inputs for negative testing."""
    return [
        {'gender': 'Invalid'},  # Invalid gender
        {'SeniorCitizen': 2},  # Invalid senior citizen value
        {'tenure': -1},  # Negative tenure
        {'MonthlyCharges': -50},  # Negative charges
        {},  # Empty dict
    ]