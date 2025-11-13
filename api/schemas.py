"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Optional, Literal
from enum import Enum


class Gender(str, Enum):
    """Gender options"""
    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    """Yes/No options"""
    YES = "Yes"
    NO = "No"


class MultipleLinesOption(str, Enum):
    """Multiple lines options"""
    YES = "Yes"
    NO = "No"
    NO_PHONE = "No phone service"


class InternetServiceOption(str, Enum):
    """Internet service options"""
    DSL = "DSL"
    FIBER = "Fiber optic"
    NO = "No"


class InternetDependentOption(str, Enum):
    """Options for internet-dependent services"""
    YES = "Yes"
    NO = "No"
    NO_INTERNET = "No internet service"


class ContractType(str, Enum):
    """Contract types"""
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethodOption(str, Enum):
    """Payment method options"""
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """
    Schema for customer features input.
    Validates all input data for prediction.
    """
    
    gender: Gender = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is senior citizen (0 or 1)")
    Partner: YesNo = Field(..., description="Whether customer has a partner")
    Dependents: YesNo = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=100, description="Number of months with company")
    PhoneService: YesNo = Field(..., description="Whether customer has phone service")
    MultipleLines: MultipleLinesOption = Field(..., description="Whether customer has multiple lines")
    InternetService: InternetServiceOption = Field(..., description="Type of internet service")
    OnlineSecurity: InternetDependentOption = Field(..., description="Whether customer has online security")
    OnlineBackup: InternetDependentOption = Field(..., description="Whether customer has online backup")
    DeviceProtection: InternetDependentOption = Field(..., description="Whether customer has device protection")
    TechSupport: InternetDependentOption = Field(..., description="Whether customer has tech support")
    StreamingTV: InternetDependentOption = Field(..., description="Whether customer has streaming TV")
    StreamingMovies: InternetDependentOption = Field(..., description="Whether customer has streaming movies")
    Contract: ContractType = Field(..., description="Contract type")
    PaperlessBilling: YesNo = Field(..., description="Whether customer uses paperless billing")
    PaymentMethod: PaymentMethodOption = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, le=200, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    @field_validator('TotalCharges', mode='after')
    def validate_total_charges(cls, v, info: ValidationInfo):
        """Ensure TotalCharges is reasonable given tenure and MonthlyCharges"""
        other = info.data or {}
        if 'tenure' in other and 'MonthlyCharges' in other:
            tenure = other['tenure']
            monthly = other['MonthlyCharges']
            if v is not None and v > 0 and v < monthly * 0.5:
                raise ValueError("TotalCharges seems too low for the given tenure and MonthlyCharges")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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


class PredictionRequest(BaseModel):
    """Schema for single prediction request"""
    customer: CustomerFeatures


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    customers: List[CustomerFeatures] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: str = Field(..., description="Churn prediction: 'Yes' or 'No'")
    prediction_label: int = Field(..., description="Numeric prediction: 0 (No) or 1 (Yes)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    no_churn_probability: float = Field(..., ge=0, le=1, description="Probability of no churn")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of prediction")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High, or Critical")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "Yes",
                "prediction_label": 1,
                "churn_probability": 0.7245,
                "no_churn_probability": 0.2755,
                "confidence": 0.7245,
                "risk_level": "High"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "Yes",
                        "prediction_label": 1,
                        "churn_probability": 0.7245,
                        "no_churn_probability": 0.2755,
                        "confidence": 0.7245,
                        "risk_level": "High"
                    }
                ],
                "total_customers": 1,
                "high_risk_count": 1
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    api_version: str
    model_path: Optional[str] = None


class ErrorResponse(BaseModel):
    """Schema for error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Prediction Error",
                "detail": "Invalid input data format",
                "status_code": 400
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information"""
    model_name: str
    model_path: str
    model_type: str
    training_date: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "XGBoost Classifier",
                "model_path": "artifacts/models/xgboost.pkl",
                "model_type": "XGBClassifier",
                "training_date": "2025-11-12"
            }
        }