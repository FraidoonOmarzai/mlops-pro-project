"""
Script to test the FastAPI endpoints.
Run this after starting the API server.
"""

import requests
import json


def test_health():
    """Test health endpoint."""
    print("\n" + "="*50)
    print("Testing Health Endpoint")
    print("="*50)
    
    response = requests.get("http://localhost:8000/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "="*50)
    print("Testing Single Prediction Endpoint")
    print("="*50)
    
    # Sample customer data
    customer_data = {
        "customer": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 840.50
        }
    }
    
    response = requests.post(
        "http://localhost:8000/predict",
        json=customer_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*50)
    print("Testing Batch Prediction Endpoint")
    print("="*50)
    
    # Sample batch data
    batch_data = {
        "customers": [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            },
            {
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 105.50,
                "TotalCharges": 6330.00
            }
        ]
    }
    
    response = requests.post(
        "http://localhost:8000/predict/batch",
        json=batch_data
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*50)
    print("Testing Model Info Endpoint")
    print("="*50)
    
    response = requests.get("http://localhost:8000/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("API ENDPOINT TESTING")
    print("="*70)
    print("\nMake sure the API server is running: python run_api.py")
    print("="*70)
    
    try:
        # Test all endpoints
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED!")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("Please start the API server first: python run_api.py\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    main()