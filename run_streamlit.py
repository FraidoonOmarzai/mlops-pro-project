"""
Script to run the Streamlit app.
Usage: python run_streamlit.py
"""

import os
import sys
import subprocess


def main():
    """Run the Streamlit app."""
    print("="*70)
    print("STARTING CUSTOMER CHURN PREDICTION DASHBOARD")
    print("="*70)
    print("\nDashboard will open in your browser")
    print("URL: http://localhost:8501")
    print("\nFeatures:")
    print("  - Single customer prediction")
    print("  - Batch prediction from CSV")
    print("  - Interactive visualizations")
    print("  - Risk assessment")
    print("\n" + "="*70)
    print("Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app/app.py",
        "--server.port=8501",
        "--server.address=localhost"
    ])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStreamlit app stopped.")
        sys.exit(0)