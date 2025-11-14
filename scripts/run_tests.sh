#!/bin/bash

# Script to run tests with different configurations
# Usage: ./scripts/run_tests.sh [test_type]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test type (default: all)
TEST_TYPE=${1:-all}

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Running MLOps Tests${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Function to run tests
run_tests() {
    local marker=$1
    local description=$2
    
    echo -e "${YELLOW}Running ${description}...${NC}"
    pytest -m "$marker" -v
    echo ""
}

# Run tests based on type
case $TEST_TYPE in
    all)
        echo "Running all tests..."
        pytest -v --cov=src --cov=api --cov-report=html --cov-report=term
        ;;
    
    unit)
        run_tests "unit" "Unit Tests"
        ;;
    
    integration)
        run_tests "integration" "Integration Tests"
        ;;
    
    data)
        run_tests "data" "Data Quality Tests"
        ;;
    
    model)
        run_tests "model" "Model Performance Tests"
        ;;
    
    api)
        run_tests "api" "API Tests"
        ;;
    
    fast)
        echo "Running fast tests (excluding slow tests)..."
        pytest -v -m "not slow"
        ;;
    
    coverage)
        echo "Running tests with coverage report..."
        pytest -v --cov=src --cov=api \
               --cov-report=html \
               --cov-report=term-missing \
               --cov-report=xml
        echo ""
        echo "Coverage report generated:"
        echo "  - HTML: htmlcov/index.html"
        echo "  - XML: coverage.xml"
        ;;
    
    ci)
        echo "Running CI tests..."
        pytest -v --cov=src --cov=api \
               --cov-report=xml \
               --cov-fail-under=70 \
               -m "not slow and not skip_docker"
        ;;
    
    clean)
        echo "Cleaning test artifacts..."
        rm -rf .pytest_cache
        rm -rf htmlcov
        rm -rf .coverage
        rm -rf coverage.xml
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        echo -e "${GREEN}Test artifacts cleaned${NC}"
        ;;
    
    *)
        echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
        echo ""
        echo "Usage: $0 {all|unit|integration|data|model|api|fast|coverage|ci|clean}"
        echo ""
        echo "Test types:"
        echo "  all         - Run all tests"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  data        - Run data quality tests only"
        echo "  model       - Run model performance tests only"
        echo "  api         - Run API tests only"
        echo "  fast        - Run fast tests (exclude slow)"
        echo "  coverage    - Run with detailed coverage report"
        echo "  ci          - Run CI pipeline tests"
        echo "  clean       - Clean test artifacts"
        exit 1
        ;;
esac

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Tests Completed${NC}"
echo -e "${GREEN}======================================${NC}"