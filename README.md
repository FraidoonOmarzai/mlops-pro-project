# Customer Churn Prediction - MLOps Project

An end-to-end MLOps project for predicting customer churn using machine learning, with complete CI/CD pipeline, containerization, and cloud deployment.

## 📋 Project Overview

This project demonstrates a production-ready ML system with:
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Experiment Tracking**: MLflow for comprehensive tracking
- **Complete Testing**: Unit, integration, and data quality tests
- **Containerization**: Docker & Docker Hub
- **Orchestration**: Kubernetes deployment
- **CI/CD**: GitHub Actions automation
- **Cloud Deployment**: AWS (EKS, S3, ECR)
- **Monitoring**: Model performance tracking

## 🎯 Business Problem

Predict customer churn to enable proactive retention strategies, reducing customer attrition by 15% and improving customer lifetime value.

## 📊 Dataset

- **Source**: Telco Customer Churn Dataset
- **Size**: ~7000 customers
- **Features**: 20 features (demographics, services, contract details)
- **Target**: Binary classification (Churn: Yes/No)

## 🏗️ Project Structure

```
mlops-churn-prediction/
│
├── config/                  # Configuration files
│   ├── config.yaml         # Main configuration
│   └── model_config.yaml   # Model hyperparameters
│
├── data/                   # Data storage (not in git)
│   ├── raw/               # Original data
│   └── processed/         # Cleaned data
│
├── src/                   # Source code
│   ├── components/        # ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_preprocessing.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/          # ML pipelines
│   │   └── training_pipeline.py
│   │
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration manager
│   ├── logger.py          # Logging setup
│   └── exception.py       # Custom exceptions
│
├── scripts/               # Executable scripts
│   ├── download_data.py  # Download dataset
│   └── train.py          # Run training pipeline
│
├── artifacts/            # Generated artifacts (not in git)
│   ├── models/          # Trained models
│   ├── preprocessors/   # Data transformers
│   └── metrics/         # Evaluation metrics
│
├── mlruns/              # MLflow tracking (not in git)
├── logs/                # Application logs (not in git)
├── tests/               # Test suite (Phase 4)
└── requirements.txt     # Python dependencies
```

## 🚀 Phase 1: Foundation (COMPLETED)

### Features Implemented

✅ **Data Pipeline**
- Automated data ingestion with train/test split
- Schema validation and data quality checks
- Feature engineering and preprocessing
- Data transformation pipelines

✅ **Model Training**
- Multiple model training (4 algorithms)
- Hyperparameter configuration
- Stratified sampling for balanced splits
- Automated model saving

✅ **Experiment Tracking**
- MLflow integration
- Parameter logging
- Metric tracking
- Model versioning

✅ **Model Evaluation**
- Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix analysis
- Model comparison
- Best model selection

✅ **Configuration Management**
- YAML-based configuration
- Centralized config manager
- Environment-specific settings

✅ **Logging & Error Handling**
- Structured logging
- Custom exception handling
- Detailed error tracking

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Git

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd mlops-churn-prediction
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Directory Structure
```bash
# Directories will be created automatically, but you can manually create them:
mkdir -p data/raw data/processed artifacts/models artifacts/preprocessors artifacts/metrics logs mlruns
```

## 📥 Download Dataset

### Option 1: Automatic Download (Recommended)
```bash
python scripts/download_data.py
```

### Option 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download the dataset
3. Save as: `data/raw/churn_data.csv`

### Option 3: Kaggle API
```bash
# Install Kaggle
pip install kaggle

# Set up credentials (~/.kaggle/kaggle.json)
# Then run:
python scripts/download_data.py
```

## 🎯 Run Training Pipeline

### Execute Complete Pipeline
```bash
python scripts/train.py
```

### What Happens:
1. **Data Ingestion**: Loads and splits data (80/20)
2. **Data Validation**: Checks schema and quality
3. **Data Preprocessing**: Cleans and transforms features
4. **Model Training**: Trains 4 models with MLflow tracking
5. **Model Evaluation**: Compares models and selects best

### Expected Output:
```
======================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY!
======================================================================

Best Model: xgboost
Models trained: 4
Preprocessor saved at: artifacts/preprocessors/preprocessor.pkl

Check MLflow UI for detailed experiment tracking:
  Run: mlflow ui
  Open: http://localhost:5000
======================================================================
```

## 📊 View Experiments with MLflow

### Start MLflow UI
```bash
mlflow ui
```

### Access Dashboard
Open browser: http://localhost:5000

### What You'll See:
- All experiment runs
- Parameters for each model
- Metrics (accuracy, precision, recall, F1, ROC-AUC)
- Model artifacts
- Comparison charts

## 📈 Evaluation Metrics

### Model Performance Targets
- **Recall**: ≥ 80% (catch most churners)
- **Precision**: ≥ 70% (avoid false alarms)
- **F1-Score**: ≥ 0.75
- **ROC-AUC**: ≥ 0.85

### Metrics Calculated
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Specificity, Sensitivity
- Classification Report

### View Results
```bash
# Check evaluation report
cat artifacts/metrics/evaluation_report.json

# Check validation report
cat artifacts/validation_report.json
```

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)
- Data paths
- Train/test split ratio
- Feature lists
- Artifact locations
- MLflow settings

### Model Configuration (`config/model_config.yaml`)
- Hyperparameters for each model
- Algorithm-specific settings
- Training parameters

## 📝 Logs

### Log Files
Logs are saved in: `logs/`

### Log Format
```
[2024-11-04 10:30:45] INFO - ChurnPrediction - Starting training pipeline
[2024-11-04 10:30:46] INFO - ChurnPrediction - Data loaded: (7043, 21)
```

## 🧪 Generated Artifacts

### After Training:
```
artifacts/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── lightgbm.pkl
├── preprocessors/
│   ├── preprocessor.pkl
│   └── preprocessor_label_encoder.pkl
├── metrics/
│   └── evaluation_report.json
└── validation_report.json
```

## 🎓 Model Training Details

### Models Trained:
1. **Logistic Regression** (Baseline)
2. **Random Forest** (Ensemble)
3. **XGBoost** (Gradient Boosting)
4. **LightGBM** (Fast Gradient Boosting)

### Training Process:
- Stratified train/test split (80/20)
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Automated hyperparameter configuration
- MLflow tracking for all experiments

## 🔄 Next Phases

### Phase 2: API & UI Development
- [ ] FastAPI REST endpoints
- [ ] Streamlit dashboard
- [ ] Real-time predictions
- [ ] Model serving

### Phase 3: Containerization
- [ ] Dockerfile creation
- [ ] Docker Compose setup
- [ ] Push to Docker Hub

### Phase 4: Testing Suite
- [ ] Unit tests
- [ ] Integration tests
- [ ] Data quality tests
- [ ] Model performance tests

### Phase 5: CI/CD
- [ ] GitHub Actions workflows
- [ ] Automated testing
- [ ] Automated builds

### Phase 6: Cloud Deployment
- [ ] Kubernetes manifests
- [ ] AWS EKS deployment
- [ ] Monitoring setup

## 🐛 Troubleshooting

### Issue: Dataset not found
```bash
python scripts/download_data.py
```

### Issue: Import errors
```bash
pip install -r requirements.txt
```

### Issue: MLflow UI not starting
```bash
# Check if port 5000 is available
# Or specify different port:
mlflow ui --port 5001
```

### Issue: Memory errors
- Reduce dataset size for testing
- Use smaller model hyperparameters
- Increase system memory

## 📚 Documentation

### Code Documentation
- All functions have docstrings
- Type hints for better IDE support
- Comprehensive comments

### Configuration Documentation
- YAML files with inline comments
- Example configurations provided

## 🤝 Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Add logging

### Testing
- Write tests for new features
- Ensure all tests pass
- Maintain code coverage

## 📄 License

This project is for educational purposes.

## 👥 Authors

Your Name - MLOps Engineer

## 🙏 Acknowledgments

- Kaggle for the dataset
- MLflow for experiment tracking
- Scikit-learn, XGBoost, LightGBM communities

---
