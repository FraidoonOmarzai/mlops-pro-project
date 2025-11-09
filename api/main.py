"""
FastAPI Application for Churn Prediction
Main entry point for the API server.
"""

import sys
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ErrorResponse, ModelInfo
)
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logger
from src.exception import CustomException


# Global prediction pipeline instance
prediction_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the model on startup and cleans up on shutdown.
    """
    global prediction_pipeline
    
    # Startup
    logger.info("Starting API server...")
    try:
        # Load prediction pipeline
        prediction_pipeline = PredictionPipeline()
        logger.info("Prediction pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load prediction pipeline: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for predicting customer churn using machine learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API and loaded models.
    """
    try:
        model_loaded = prediction_pipeline is not None and prediction_pipeline.model is not None
        preprocessor_loaded = prediction_pipeline is not None and prediction_pipeline.preprocessor is not None
        
        return HealthResponse(
            status="healthy" if (model_loaded and preprocessor_loaded) else "unhealthy",
            model_loaded=model_loaded,
            preprocessor_loaded=preprocessor_loaded,
            api_version="1.0.0",
            model_path=prediction_pipeline.model_path if prediction_pipeline else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a churn prediction for a single customer.
    
    Args:
        request: Customer features
        
    Returns:
        Prediction result with probability and risk level
    """
    try:
        logger.info("Received prediction request")
        
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert Pydantic model to dict
        customer_data = request.customer.dict()
        
        # Make prediction
        result = prediction_pipeline.predict(customer_data)
        
        logger.info(f"Prediction completed: {result['prediction']}")
        return PredictionResponse(**result)
        
    except CustomException as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make churn predictions for multiple customers.
    
    Args:
        request: List of customer features (max 100)
        
    Returns:
        List of prediction results
    """
    try:
        logger.info(f"Received batch prediction request for {len(request.customers)} customers")
        
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert Pydantic models to dicts
        customers_data = [customer.dict() for customer in request.customers]
        
        # Make predictions
        results = prediction_pipeline.predict_batch(customers_data)
        
        # Filter out errors
        successful_predictions = [r for r in results if 'error' not in r]
        
        # Count high-risk customers
        high_risk_count = sum(
            1 for r in successful_predictions 
            if r.get('risk_level') in ['High', 'Critical']
        )
        
        logger.info(f"Batch prediction completed: {len(successful_predictions)} successful")
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in successful_predictions],
            total_customers=len(successful_predictions),
            high_risk_count=high_risk_count
        )
        
    except CustomException as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Model info endpoint
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Model information including name, path, and type
    """
    try:
        if prediction_pipeline is None or prediction_pipeline.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        model_type = type(prediction_pipeline.model).__name__
        
        return ModelInfo(
            model_name=model_type,
            model_path=prediction_pipeline.model_path,
            model_type=model_type,
            training_date=None  # Could be extracted from model metadata
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature importance endpoint
@app.get("/model/feature-importance", tags=["Model"])
async def get_feature_importance():
    """
    Get feature importance from the model.
    
    Returns:
        Dictionary of feature names and their importance scores
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        importance = prediction_pipeline.get_feature_importance()
        
        if not importance:
            return {
                "message": "Feature importance not available for this model type",
                "feature_importance": {}
            }
        
        # Return top 20 features
        top_features = dict(list(importance.items())[:20])
        
        return {
            "feature_importance": top_features,
            "total_features": len(importance)
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoint
@app.get("/stats", tags=["Statistics"])
async def get_statistics():
    """
    Get API usage statistics.
    
    Returns:
        Statistics about API usage
    """
    return {
        "message": "Statistics endpoint - To be implemented with request tracking",
        "total_predictions": 0,
        "average_response_time": 0,
        "high_risk_percentage": 0
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )