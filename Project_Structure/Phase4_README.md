# Phase 4: Testing Suite - Complete Guide

## 🧪 Overview

Phase 4 implements a comprehensive testing framework covering unit tests, integration tests, data validation, model performance, and API testing.

---

## 📦 What's Been Created

### **Test Structure (4 Categories)**
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Pipeline and API testing
3. **Data Tests** - Data quality and validation
4. **Model Tests** - Performance and fairness testing

### **Test Files (11)**
1. `pytest.ini` - Pytest configuration
2. `coveragerc` - Coverage settings
3. `tests/conftest.py` - Shared fixtures
4. `tests/unit/test_data_ingestion.py`
5. `tests/unit/test_prediction_pipeline.py`
6. `tests/integration/test_api_endpoints.py`
7. `tests/data/test_data_quality.py`
8. `tests/model/test_model_performance.py`
9. `scripts/run_tests.sh` - Test runner (Linux/Mac)
10. `scripts/run_tests.bat` - Test runner (Windows)
11. This README

---

## 🚀 Quick Start

### **Step 1: Install Test Dependencies**
```bash
pip install pytest pytest-cov pytest-mock pytest-timeout
```

Or use existing requirements.txt:
```bash
pip install -r requirements.txt
```

### **Step 2: Run All Tests**
```bash
# Linux/Mac
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh all

# Windows
scripts\run_tests.bat all

# Or directly with pytest
pytest -v
```

### **Step 3: View Coverage Report**
```bash
./scripts/run_tests.sh coverage

# Then open in browser
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov\index.html  # Windows
```

---

## 📊 Test Categories

### **1. Unit Tests** (`tests/unit/`)
Test individual components in isolation.

**Run:**
```bash
pytest -m unit -v
./scripts/run_tests.sh unit
```

**Tests:**
- Data ingestion logic
- Data preprocessing
- Model training components
- Prediction pipeline
- Utility functions

**Example:**
```bash
pytest tests/unit/test_data_ingestion.py -v
```

### **2. Integration Tests** (`tests/integration/`)
Test complete workflows and API endpoints.

**Run:**
```bash
pytest -m integration -v
./scripts/run_tests.sh integration
```

**Tests:**
- End-to-end training pipeline
- API endpoint responses
- Service communication
- Complete prediction workflow

**Example:**
```bash
pytest tests/integration/test_api_endpoints.py -v
```

### **3. Data Quality Tests** (`tests/data/`)
Validate data quality and integrity.

**Run:**
```bash
pytest -m data -v
./scripts/run_tests.sh data
```

**Tests:**
- Missing values
- Data types
- Valid categories
- Numerical ranges
- Data consistency
- Class distribution

**Example:**
```bash
pytest tests/data/test_data_quality.py -v
```

### **4. Model Performance Tests** (`tests/model/`)
Ensure model meets performance requirements.

**Run:**
```bash
pytest -m model -v
./scripts/run_tests.sh model
```

**Tests:**
- Minimum accuracy (60%)
- Minimum precision (50%)
- Minimum recall (50%)
- Minimum F1 score (55%)
- Minimum ROC-AUC (70%)
- No constant predictions
- Fairness across groups

**Example:**
```bash
pytest tests/model/test_model_performance.py -v
```

---

## 🎯 Test Markers

Tests are organized using pytest markers:

```bash
# Run specific marker
pytest -m unit
pytest -m integration
pytest -m data
pytest -m model
pytest -m api
pytest -m slow
pytest -m requires_model

# Combine markers
pytest -m "unit and not slow"
pytest -m "integration or api"

# Exclude markers
pytest -m "not slow"
```

---

## 📈 Coverage Requirements

### **Minimum Coverage: 70%**

Check coverage:
```bash
pytest --cov=src --cov=api --cov-report=term-missing
```

### **Coverage Reports Generated:**
1. **Terminal** - Summary in console
2. **HTML** - Detailed report in `htmlcov/`
3. **XML** - For CI/CD integration

### **View HTML Report:**
```bash
# Generate report
pytest --cov=src --cov=api --cov-report=html

# Open in browser
open htmlcov/index.html
```

---

## 🔧 Test Configuration

### **pytest.ini**
```ini
[pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    data: Data quality tests
    model: Model performance tests
    api: API tests
    slow: Slow tests
addopts = -v --strict-markers --cov-fail-under=70
```

### **.coveragerc**
```ini
[run]
source = src,api
omit = */tests/*, */venv/*

[report]
precision = 2
show_missing = True
```

---

## 🧩 Test Fixtures

Located in `tests/conftest.py`, available to all tests:

### **Configuration Fixtures**
- `test_config` - Test configuration
- `temp_dir` - Temporary directory

### **Data Fixtures**
- `sample_customer_data` - Single customer dict
- `sample_dataframe` - DataFrame with 5 customers
- `sample_csv_file` - Temporary CSV file

### **Model Fixtures**
- `mock_model` - Trained model mock
- `mock_preprocessor` - Preprocessor mock
- `prediction_pipeline_mock` - Complete pipeline mock

### **API Fixtures**
- `api_client` - FastAPI test client

### **Utility Fixtures**
- `mock_logger` - Suppress log output
- `mock_mlflow` - Mock MLflow tracking

---

## 🎨 Writing New Tests

### **Example Unit Test**
```python
import pytest

@pytest.mark.unit
def test_my_function(sample_dataframe):
    """Test my function works correctly."""
    result = my_function(sample_dataframe)
    assert result is not None
    assert len(result) > 0
```

### **Example Integration Test**
```python
@pytest.mark.integration
@pytest.mark.api
def test_api_endpoint(api_client):
    """Test API endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
```

### **Example Parametrized Test**
```python
@pytest.mark.parametrize("input,expected", [
    (0.1, 'Low'),
    (0.5, 'High'),
    (0.8, 'Critical'),
])
def test_risk_levels(input, expected):
    """Test risk level calculation."""
    assert calculate_risk(input) == expected
```

---

## 📋 Test Runner Commands

### **Basic Commands**
```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run specific file
pytest tests/unit/test_data_ingestion.py

# Run specific test
pytest tests/unit/test_data_ingestion.py::test_function_name
```

### **Script Commands**
```bash
./scripts/run_tests.sh all        # All tests
./scripts/run_tests.sh unit       # Unit tests
./scripts/run_tests.sh integration # Integration tests
./scripts/run_tests.sh data       # Data tests
./scripts/run_tests.sh model      # Model tests
./scripts/run_tests.sh api        # API tests
./scripts/run_tests.sh fast       # Exclude slow tests
./scripts/run_tests.sh coverage   # With coverage
./scripts/run_tests.sh ci         # CI pipeline
./scripts/run_tests.sh clean      # Clean artifacts
```

---

## 🔍 Debugging Tests

### **Run with Debug Info**
```bash
pytest -vv --tb=long
```

### **Show Print Statements**
```bash
pytest -s
```

### **Drop into Debugger on Failure**
```bash
pytest --pdb
```

### **Show Fixtures**
```bash
pytest --fixtures
```

### **Collect Tests Without Running**
```bash
pytest --collect-only
```

---

## 🚀 CI/CD Integration

### **GitHub Actions Example**
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=src --cov=api --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 📊 Performance Testing

### **Test Execution Time**
```bash
# Show slowest tests
pytest --durations=10

# Set timeout
pytest --timeout=60
```

### **Parallel Execution**
```bash
# Install plugin
pip install pytest-xdist

# Run in parallel
pytest -n auto
```

---

## 🐛 Common Issues

### **Issue: Tests fail with import errors**
```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### **Issue: Fixtures not found**
Ensure `conftest.py` is in the `tests/` directory.

### **Issue: Coverage too low**
```bash
# See missing lines
pytest --cov=src --cov-report=term-missing

# Focus on specific modules
pytest --cov=src.components
```

### **Issue: Tests run slow**
```bash
# Skip slow tests
pytest -m "not slow"

# Use parallel execution
pytest -n auto
```

---

## ✅ Phase 4 Checklist

- ✅ Pytest configuration (`pytest.ini`)
- ✅ Coverage configuration (`.coveragerc`)
- ✅ Test fixtures (`conftest.py`)
- ✅ Unit tests (data, models, pipelines)
- ✅ Integration tests (API, end-to-end)
- ✅ Data quality tests
- ✅ Model performance tests
- ✅ Test runners (bash & batch scripts)
- ✅ Coverage reporting (HTML, XML, terminal)
- ✅ Test markers and organization
- ✅ Documentation

---

## 🎯 Success Criteria

Phase 4 is complete when:
1. ✅ All test categories implemented
2. ✅ Test coverage ≥ 70%
3. ✅ All tests pass
4. ✅ Test runners work on all platforms
5. ✅ Coverage reports generate correctly
6. ✅ Tests run in < 5 minutes
7. ✅ CI/CD ready

---

## 📚 Best Practices

1. **Write tests first** (TDD approach)
2. **Keep tests independent**
3. **Use descriptive test names**
4. **One assertion per test** (when possible)
5. **Use fixtures for setup**
6. **Mock external dependencies**
7. **Test edge cases**
8. **Maintain fast test suite**

---

## 🔄 Next Steps (Phase 5: CI/CD)

Once testing is complete:
- ✅ GitHub Actions workflows
- ✅ Automated testing on push
- ✅ Automated Docker builds
- ✅ Code quality checks
- ✅ Security scanning
- ✅ Automated deployment

---

**Phase 4 Complete! Ready for Phase 5?** 🚀