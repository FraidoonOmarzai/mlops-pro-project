@echo off
REM Script to run tests on Windows
REM Usage: scripts\run_tests.bat [test_type]

echo ======================================
echo Running MLOps Tests
echo ======================================
echo.

set TEST_TYPE=%1
if "%TEST_TYPE%"=="" set TEST_TYPE=all

if "%TEST_TYPE%"=="all" (
    echo Running all tests...
    pytest -v --cov=src --cov=api --cov-report=html --cov-report=term
    goto :end
)

if "%TEST_TYPE%"=="unit" (
    echo Running Unit Tests...
    pytest -m unit -v
    goto :end
)

if "%TEST_TYPE%"=="integration" (
    echo Running Integration Tests...
    pytest -m integration -v
    goto :end
)

if "%TEST_TYPE%"=="data" (
    echo Running Data Quality Tests...
    pytest -m data -v
    goto :end
)

if "%TEST_TYPE%"=="model" (
    echo Running Model Performance Tests...
    pytest -m model -v
    goto :end
)

if "%TEST_TYPE%"=="api" (
    echo Running API Tests...
    pytest -m api -v
    goto :end
)

if "%TEST_TYPE%"=="fast" (
    echo Running fast tests...
    pytest -v -m "not slow"
    goto :end
)

if "%TEST_TYPE%"=="coverage" (
    echo Running tests with coverage...
    pytest -v --cov=src --cov=api --cov-report=html --cov-report=term-missing --cov-report=xml
    echo.
    echo Coverage report generated:
    echo   - HTML: htmlcov\index.html
    echo   - XML: coverage.xml
    goto :end
)

if "%TEST_TYPE%"=="clean" (
    echo Cleaning test artifacts...
    if exist .pytest_cache rmdir /s /q .pytest_cache
    if exist htmlcov rmdir /s /q htmlcov
    if exist .coverage del /q .coverage
    if exist coverage.xml del /q coverage.xml
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    echo Test artifacts cleaned
    goto :end
)

echo Invalid test type: %TEST_TYPE%
echo.
echo Usage: %0 {all^|unit^|integration^|data^|model^|api^|fast^|coverage^|clean}
exit /b 1

:end
echo.
echo ======================================
echo Tests Completed
echo ======================================