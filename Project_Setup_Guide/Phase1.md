# Phase 1 - Quick Setup Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Create Project Structure
```bash
# Create main directory
mkdir mlops-churn-prediction
cd mlops-churn-prediction

# Create subdirectories
mkdir -p config data/raw data/processed src/components src/pipeline src/utils scripts artifacts/models artifacts/preprocessors artifacts/metrics logs tests
```

### Step 2: Create Empty __init__.py Files
```bash
# Windows
type nul > src/__init__.py
type nul > src/components/__init__.py
type nul > src/pipeline/__init__.py
type nul > src/utils/__init__.py
type nul > tests/__init__.py

# Linux/Mac
touch src/__init__.py
touch src/components/__init__.py
touch src/pipeline/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

### Step 3: Copy All Files
Copy the following files from the artifacts I created:

**Configuration:**
- `config/config.yaml`
- `config/model_config.yaml`

**Core Utilities:**
- `src/logger.py`
- `src/exception.py`
- `src/config.py`
- `src/utils/common.py`

**Components:**
- `src/components/data_ingestion.py`
- `src/components/data_validation.py`
- `src/components/data_preprocessing.py`
- `src/components/model_trainer.py`
- `src/components/model_evaluation.py`

**Pipeline:**
- `src/pipeline/training_pipeline.py`

**Scripts:**
- `scripts/download_data.py`
- `scripts/train.py`

**Setup Files:**
- `requirements.txt`
- `README.md`
- `.gitignore`

### Step 4: Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download Dataset
```bash
python scripts/download_data.py
```

### Step 6: Run Training Pipeline
```bash
python scripts/train.py
```

### Step 7: View Results with MLflow
```bash
mlflow ui
```
Then open: http://localhost:5000

---

## 📋 File Checklist

Make sure you have all these files:

```
✅ config/config.yaml
✅ config/model_config.yaml
✅ src/__init__.py
✅ src/logger.py
✅ src/exception.py
✅ src/config.py
✅ src/components/__init__.py
✅ src/components/data_ingestion.py
✅ src/components/data_validation.py
✅ src/components/data_preprocessing.py
✅ src/components/model_trainer.py
✅ src/components/model_evaluation.py
✅ src/pipeline/__init__.py
✅ src/pipeline/training_pipeline.py
✅ src/utils/__init__.py
✅ src/utils/common.py
✅ scripts/download_data.py
✅ scripts/train.py
✅ requirements.txt
✅ README.md
✅ .gitignore
```

---

## 🔍 Verify Installation

### Check Python Version
```bash
python --version
# Should be 3.8 or higher
```

### Check Virtual Environment
```bash
which python  # Linux/Mac
where python  # Windows
# Should point to venv directory
```

### Test Imports
```bash
python -c "import pandas, numpy, sklearn, mlflow; print('All imports successful!')"
```

---

## 📊 Expected Results

After running `python scripts/train.py`, you should see:

1. **Console Output:**
   - Data ingestion logs
   - Validation results
   - Preprocessing steps
   - Training progress for 4 models
   - Evaluation metrics
   - Best model selection

2. **Generated Files:**
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
   
   mlruns/
   └── 0/  # Experiment runs
   
   logs/
   └── 2024_11_04_*.log
   
   data/
   ├── raw/
   │   └── churn_data.csv
   └── processed/
       ├── train.csv
       └── test.csv
   ```

3. **MLflow UI:**
   - 4 experiment runs (one per model)
   - Metrics comparison
   - Model artifacts

---

## 🐛 Common Issues & Solutions

### Issue 1: Module not found error
```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows
```

### Issue 2: Data file not found
```bash
# Solution: Verify data path
ls data/raw/churn_data.csv  # Linux/Mac
dir data\raw\churn_data.csv  # Windows
```

### Issue 3: MLflow UI won't start
```bash
# Solution: Use different port
mlflow ui --port 5001
```

### Issue 4: Permission denied
```bash
# Solution: Check directory permissions
chmod +x scripts/*.py  # Linux/Mac
```

---

## 🎯 Phase 1 Success Criteria

✅ All dependencies installed
✅ Dataset downloaded successfully
✅ Training pipeline executes without errors
✅ 4 models trained successfully
✅ MLflow tracks all experiments
✅ Artifacts generated in correct locations
✅ Best model selected and saved
✅ Evaluation report generated

---

## 📝 Next Steps

Once Phase 1 is complete and verified:

1. ✅ Review the evaluation report
2. ✅ Check MLflow UI for experiment details
3. ✅ Verify all artifacts are generated
4. ✅ **Confirm with me to proceed to Phase 2**

---

## 💡 Tips

1. **Keep your virtual environment activated** while working
2. **Check logs/** directory if something fails
3. **Use MLflow UI** to compare model performances
4. **Review artifacts/** to see generated files
5. **Read evaluation_report.json** for detailed metrics

---

## 🆘 Need Help?

If you encounter issues:

1. Check the logs in `logs/` directory
2. Verify all files are in correct locations
3. Ensure virtual environment is activated
4. Check Python version (3.8+)
5. Confirm all dependencies installed