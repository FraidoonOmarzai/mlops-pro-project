@echo off
REM Script to push Docker images to Docker Hub on Windows
REM Usage: scripts\push_images.bat [version]

echo ======================================
echo Pushing Docker Images to Docker Hub
echo ======================================
echo.

REM Load environment variables
if exist .env (
    for /f "tokens=*" %%a in (.env) do (
        set "%%a"
    )
) else (
    echo Error: .env file not found
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

REM Check if logged in
echo Checking Docker Hub login...
docker info | find "Username" >nul
if errorlevel 1 (
    echo Please login to Docker Hub
    docker login
)

REM Push API image
echo [1/3] Pushing API image...
docker push %DOCKER_USERNAME%/churn-prediction-api:%IMAGE_VERSION%
docker push %DOCKER_USERNAME%/churn-prediction-api:latest
if errorlevel 1 (
    echo Failed to push API image
    exit /b 1
)
echo API image pushed successfully
echo.

REM Push Streamlit image
echo [2/3] Pushing Streamlit image...
docker push %DOCKER_USERNAME%/churn-prediction-streamlit:%IMAGE_VERSION%
docker push %DOCKER_USERNAME%/churn-prediction-streamlit:latest
if errorlevel 1 (
    echo Failed to push Streamlit image
    exit /b 1
)
echo Streamlit image pushed successfully
echo.

REM Push Training image
echo [3/3] Pushing Training image...
docker push %DOCKER_USERNAME%/churn-prediction-training:%IMAGE_VERSION%
docker push %DOCKER_USERNAME%/churn-prediction-training:latest
if errorlevel 1 (
    echo Failed to push Training image
    exit /b 1
)
echo Training image pushed successfully
echo.

echo ======================================
echo Push Summary
echo ======================================
echo Images pushed to Docker Hub:
echo   - %DOCKER_USERNAME%/churn-prediction-api:%IMAGE_VERSION%
echo   - %DOCKER_USERNAME%/churn-prediction-streamlit:%IMAGE_VERSION%
echo   - %DOCKER_USERNAME%/churn-prediction-training:%IMAGE_VERSION%
echo.
echo View your images at:
echo   https://hub.docker.com/u/%DOCKER_USERNAME%
echo.
echo Images are now available for deployment!
echo.