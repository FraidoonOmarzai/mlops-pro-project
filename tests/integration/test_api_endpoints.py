"""
Integration tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
@pytest.mark.api
class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""
    
    def test_root_endpoint(self, api_client):
        """Test root endpoint returns API information."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data
        assert "api_version" in data
    
    def test_predict_endpoint_valid_input(self, api_client, sample_customer_data):
        """Test prediction endpoint with valid input."""
        payload = {"customer": sample_customer_data}
        
        response = api_client.post("/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "prediction" in data
            assert "prediction_label" in data
            assert "churn_probability" in data
            assert "confidence" in data
            assert "risk_level" in data
            
            assert data["prediction"] in ["Yes", "No"]
            assert data["prediction_label"] in [0, 1]
            assert 0 <= data["churn_probability"] <= 1
            assert 0 <= data["confidence"] <= 1
            assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
    
    def test_predict_endpoint_invalid_input(self, api_client):
        """Test prediction endpoint with invalid input."""
        invalid_payload = {"customer": {"invalid": "data"}}
        
        response = api_client.post("/predict", json=invalid_payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self, api_client):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "customer": {
                "gender": "Male",
                "tenure": 12
                # Missing other required fields
            }
        }
        
        response = api_client.post("/predict", json=incomplete_data)
        
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self, api_client, valid_customer_inputs):
        """Test batch prediction endpoint."""
        payload = {"customers": valid_customer_inputs}
        
        response = api_client.post("/predict/batch", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "predictions" in data
            assert "total_customers" in data
            assert "high_risk_count" in data
            
            assert len(data["predictions"]) <= len(valid_customer_inputs)
            assert isinstance(data["total_customers"], int)
            assert isinstance(data["high_risk_count"], int)
    
    def test_batch_predict_empty_list(self, api_client):
        """Test batch prediction with empty list."""
        payload = {"customers": []}
        
        response = api_client.post("/predict/batch", json=payload)
        
        assert response.status_code == 422  # Validation error (min_items=1)
    
    def test_batch_predict_too_many_customers(self, api_client, sample_customer_data):
        """Test batch prediction with too many customers."""
        # Create more than 100 customers
        customers = [sample_customer_data] * 101
        payload = {"customers": customers}
        
        response = api_client.post("/predict/batch", json=payload)
        
        assert response.status_code == 422  # Exceeds max_items
    
    def test_model_info_endpoint(self, api_client):
        """Test model info endpoint."""
        response = api_client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            
            assert "model_name" in data
            assert "model_path" in data
            assert "model_type" in data
    
    def test_feature_importance_endpoint(self, api_client):
        """Test feature importance endpoint."""
        response = api_client.get("/model/feature-importance")
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "feature_importance" in data or "message" in data
    
    def test_stats_endpoint(self, api_client):
        """Test statistics endpoint."""
        response = api_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data or "total_predictions" in data


@pytest.mark.integration
@pytest.mark.api
class TestAPIValidation:
    """Test API input validation."""
    
    def test_invalid_gender(self, api_client, sample_customer_data):
        """Test validation rejects invalid gender."""
        data = sample_customer_data.copy()
        data['gender'] = 'InvalidGender'
        
        response = api_client.post("/predict", json={"customer": data})
        assert response.status_code == 422
    
    def test_invalid_senior_citizen(self, api_client, sample_customer_data):
        """Test validation rejects invalid SeniorCitizen value."""
        data = sample_customer_data.copy()
        data['SeniorCitizen'] = 5  # Should be 0 or 1
        
        response = api_client.post("/predict", json={"customer": data})
        assert response.status_code == 422
    
    def test_negative_tenure(self, api_client, sample_customer_data):
        """Test validation rejects negative tenure."""
        data = sample_customer_data.copy()
        data['tenure'] = -5
        
        response = api_client.post("/predict", json={"customer": data})
        assert response.status_code == 422
    
    def test_negative_charges(self, api_client, sample_customer_data):
        """Test validation rejects negative charges."""
        data = sample_customer_data.copy()
        data['MonthlyCharges'] = -50.0
        
        response = api_client.post("/predict", json={"customer": data})
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.slow
def test_api_response_time(api_client, sample_customer_data):
    """Test API response time is reasonable."""
    import time
    
    payload = {"customer": sample_customer_data}
    
    start_time = time.time()
    response = api_client.post("/predict", json=payload)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Response should be under 1 second
    assert response_time < 1.0
    assert response.status_code in [200, 503]


# @pytest.mark.integration
# @pytest.mark.api
# def test_cors_headers(api_client):
#     """Test CORS headers are present."""
#     response = api_client.get("/health")
    
#     # Check for CORS headers
#     assert "access-control-allow-origin" in response.headers
