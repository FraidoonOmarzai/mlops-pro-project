# MLOps Churn Prediction - Project Structure

```
mlops-churn-prediction/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore file
│
├── config/
│   ├── config.yaml                     # Main configuration file
│   └── model_config.yaml               # Model hyperparameters
│
├── data/
│   ├── raw/                            # Original data (not in git)
│   ├── processed/                      # Cleaned data
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb    # Feature exploration
│   └── 03_model_experiments.ipynb      # Model experimentation
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration loader
│   ├── logger.py                       # Logging setup
│   ├── exception.py                    # Custom exceptions
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          # Load and split data
│   │   ├── data_validation.py         # Schema & quality checks
│   │   ├── data_preprocessing.py      # Feature engineering
│   │   ├── model_trainer.py           # Train ML models
│   │   └── model_evaluation.py        # Evaluate & compare models
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py       # Orchestrate training
│   │   └── prediction_pipeline.py     # Inference pipeline
│   │
│   └── utils/
│       ├── __init__.py
│       └── common.py                   # Helper functions
│
├── mlruns/                             # MLflow tracking (not in git)
├── artifacts/                          # Model artifacts (not in git)
│   ├── models/
│   ├── preprocessors/
│   └── metrics/
│
├── tests/                              # Tests (Phase 4)
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── data/
│
├── logs/                               # Application logs (not in git)
│
└── scripts/
    ├── download_data.py                # Download dataset
    └── train.py                        # Training entry point
```

## 📦 Files We'll Create in Phase 1

1. **Configuration Files:**
   - `config/config.yaml` - Paths, parameters
   - `config/model_config.yaml` - Model hyperparameters

2. **Core Modules:**
   - `src/config.py` - Load configurations
   - `src/logger.py` - Logging setup
   - `src/exception.py` - Exception handling

3. **Components:**
   - `src/components/data_ingestion.py`
   - `src/components/data_validation.py`
   - `src/components/data_preprocessing.py`
   - `src/components/model_trainer.py`
   - `src/components/model_evaluation.py`

4. **Pipelines:**
   - `src/pipeline/training_pipeline.py`

5. **Utilities:**
   - `src/utils/common.py`
   - `scripts/download_data.py`
   - `scripts/train.py`

6. **Setup Files:**
   - `requirements.txt`
   - `README.md`
   - `.gitignore`