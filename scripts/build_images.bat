@echo off
REM Script to build Docker images on Windows
REM Usage: scripts\build_images.bat [version]

echo ======================================
echo Building Docker Images
echo ======================================
echo.

REM Load environment variables
if exist .env (
    echo Loading environment variables from .env
    for /f "tokens=*" %%a in (.env) do (
        set "%%a"
    )
) else (
    echo Error: .env file not found
    echo Please copy .env.example to .env and configure it
    exit /b 1
)

REM Set version
if "%~1"=="" (
    set IMAGE_VERSION=%VERSION%
    if "%VERSION%"=="" set IMAGE_VERSION=latest
) else (
    set IMAGE_VERSION=%~1
)

echo Version: %IMAGE_VERSION%
echo Docker Username: %DOCKER_USERNAME%
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    exit /b 1
)

REM Build API image
echo [1/3] Building API image...
docker build -t %DOCKER_USERNAME%/churn-prediction-api:%IMAGE_VERSION% -t %DOCKER_USERNAME%/churn-prediction-api:latest -f docker/Dockerfile.api .
if errorlevel 1 (
    echo Failed to build API image
    exit /b 1
)
echo API image built successfully
echo.

REM Build Streamlit image
echo [2/3] Building Streamlit image...
docker build -t %DOCKER_USERNAME%/churn-prediction-streamlit:%IMAGE_VERSION% -t %DOCKER_USERNAME%/churn-prediction-streamlit:latest -f docker/Dockerfile.streamlit .
if errorlevel 1 (
    echo Failed to build Streamlit image
    exit /b 1
)
echo Streamlit image built successfully
echo.

REM Build Training image
echo [3/3] Building Training image...
docker build -t %DOCKER_USERNAME%/churn-prediction-training:%IMAGE_VERSION% -t %DOCKER_USERNAME%/churn-prediction-training:latest -f docker/Dockerfile.training .
if errorlevel 1 (
    echo Failed to build Training image
    exit /b 1
)
echo Training image built successfully
echo.

echo ======================================
echo Build Summary
echo ======================================
echo Images built:
echo   - %DOCKER_USERNAME%/churn-prediction-api:%IMAGE_VERSION%
echo   - %DOCKER_USERNAME%/churn-prediction-streamlit:%IMAGE_VERSION%
echo   - %DOCKER_USERNAME%/churn-prediction-training:%IMAGE_VERSION%
echo.
echo Next steps:
echo   1. Test images: docker-compose up
echo   2. Push to Docker Hub: scripts\push_images.bat
echo.