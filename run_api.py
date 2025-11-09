"""
Script to run the FastAPI server.
Usage: python run_api.py
"""

import os
import sys
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Run the FastAPI server."""
    print("="*70)
    print("STARTING CUSTOMER CHURN PREDICTION API")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  - Root:              http://localhost:8000")
    print("  - Health Check:      http://localhost:8000/health")
    print("  - Single Prediction: http://localhost:8000/predict")
    print("  - Batch Prediction:  http://localhost:8000/predict/batch")
    print("  - Model Info:        http://localhost:8000/model/info")
    print("  - API Docs:          http://localhost:8000/docs")
    print("  - ReDoc:             http://localhost:8000/redoc")
    print("\n" + "="*70)
    print("Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAPI server stopped.")
        sys.exit(0)