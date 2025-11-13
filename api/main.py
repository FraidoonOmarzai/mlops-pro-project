"""
api/main.py
Simplified FastAPI app for Customer Churn Prediction.
"""

from typing import Any, Dict, List, Optional
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ensure project root is importable (keep if you need local imports)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your pydantic schemas and pipeline
from api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfo,
)
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logger
from src.exception import CustomException

# ---------- Global state ----------
prediction_pipeline: Optional[PredictionPipeline] = None
API_VERSION = "1.0.0"


# ---------- Lifespan (startup / shutdown) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction pipeline on startup and clean up on shutdown."""
    global prediction_pipeline
    logger.info("API starting up...")
    try:
        prediction_pipeline = PredictionPipeline()
        logger.info("Prediction pipeline loaded")
    except Exception as exc:
        logger.exception("Failed to initialize prediction pipeline")
        # Re-raise so FastAPI fails to start (visible in logs)
        raise

    yield

    logger.info("API shutting down...")


# ---------- App creation ----------
app = FastAPI(
    title="Customer Churn Prediction API",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for development (lock down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Middleware: processing time ----------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.4f}"
    return response


# ---------- Global exception handler ----------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# ---------- Helpers ----------
def ensure_pipeline_loaded():
    """Raise HTTPException(503) if the pipeline is not ready."""
    if prediction_pipeline is None or getattr(prediction_pipeline, "model", None) is None:
        logger.warning("Request received but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")


# ---------- Routes ----------
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    return {
        "message": "Customer Churn Prediction API",
        "version": API_VERSION,
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    try:
        model_loaded = prediction_pipeline is not None and getattr(prediction_pipeline, "model", None) is not None
        preprocessor_loaded = prediction_pipeline is not None and getattr(prediction_pipeline, "preprocessor", None) is not None

        return HealthResponse(
            status="healthy" if model_loaded and preprocessor_loaded else "unhealthy",
            model_loaded=model_loaded,
            preprocessor_loaded=preprocessor_loaded,
            api_version=API_VERSION,
            model_path=getattr(prediction_pipeline, "model_path", None),
        )
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Single customer prediction.
    """
    ensure_pipeline_loaded()

    try:
        customer = request.customer.dict()
        result = prediction_pipeline.predict(customer)
        return PredictionResponse(**result)
    except CustomException as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch predictions (list of customers).
    """
    ensure_pipeline_loaded()

    try:
        customers = [c.dict() for c in request.customers]
        results = prediction_pipeline.predict_batch(customers)

        # keep successful predictions (no 'error' key)
        successful = [r for r in results if "error" not in r]

        high_risk_count = sum(1 for r in successful if r.get("risk_level") in {"High", "Critical"})

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in successful],
            total_customers=len(successful),
            high_risk_count=high_risk_count,
        )
    except CustomException as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unexpected batch prediction error")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info() -> ModelInfo:
    ensure_pipeline_loaded()

    model_obj = prediction_pipeline.model
    model_type = type(model_obj).__name__

    return ModelInfo(
        model_name=model_type,
        model_path=getattr(prediction_pipeline, "model_path", None),
        model_type=model_type,
        training_date=None,
    )


@app.get("/model/feature-importance", tags=["Model"])
async def get_feature_importance() -> Dict[str, Any]:
    ensure_pipeline_loaded()
    importance = prediction_pipeline.get_feature_importance() or {}

    # return top 20 features
    top = dict(list(importance.items())[:20])
    return {"feature_importance": top, "total_features": len(importance)}


@app.get("/stats", tags=["Statistics"])
async def get_statistics() -> Dict[str, Any]:
    # placeholder — implement real tracking if desired
    return {
        "message": "Statistics endpoint - To be implemented",
        "total_predictions": 0,
        "average_response_time": 0,
        "high_risk_percentage": 0,
    }


# ---------- Uvicorn entry (for local dev) ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
