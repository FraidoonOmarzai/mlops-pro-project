@echo off
REM Deploy to Local Kubernetes on Windows
REM Usage: scripts\deploy-local.bat YOUR_DOCKERHUB_USERNAME

echo =========================================
echo Deploying to Local Kubernetes (Windows)
echo =========================================
echo.

if "%1"=="" (
    echo Error: Docker Hub username required
    echo Usage: scripts\deploy-local.bat YOUR_DOCKERHUB_USERNAME
    exit /b 1
)

set DOCKER_USERNAME=%1

REM Check if minikube is running
echo Checking minikube status...
minikube status >nul 2>&1
if errorlevel 1 (
    echo Starting minikube...
    minikube start
    if errorlevel 1 (
        echo Error: Failed to start minikube
        exit /b 1
    )
)

echo Minikube is running!
echo.

REM Create namespace
echo Creating namespace...
kubectl apply -f k8s\namespace.yaml

REM Create temp directory
if not exist k8s\temp mkdir k8s\temp

REM Update API YAML with Docker username
echo Updating API configuration...
powershell -Command "(gc k8s\api.yaml) -replace 'YOUR_DOCKERHUB_USERNAME', '%DOCKER_USERNAME%' | Out-File -encoding ASCII k8s\temp\api.yaml"

REM Update Streamlit YAML with Docker username
echo Updating Streamlit configuration...
powershell -Command "(gc k8s\streamlit.yaml) -replace 'YOUR_DOCKERHUB_USERNAME', '%DOCKER_USERNAME%' | Out-File -encoding ASCII k8s\temp\streamlit.yaml"

REM Deploy API
echo Deploying API...
kubectl apply -f k8s\temp\api.yaml

REM Deploy Streamlit
echo Deploying Streamlit...
kubectl apply -f k8s\temp\streamlit.yaml

REM Wait for deployments
echo.
echo Waiting for pods to be ready (this may take 2-3 minutes)...
kubectl wait --for=condition=available --timeout=300s deployment/api -n churn-prediction 2>nul
kubectl wait --for=condition=available --timeout=300s deployment/streamlit -n churn-prediction 2>nul

echo.
echo =========================================
echo Deployment Complete!
echo =========================================
echo.

REM Get service URLs
echo To access your services, run these commands:
echo.
echo For API:
echo   minikube service api-service -n churn-prediction
echo.
echo For Streamlit:
echo   minikube service streamlit-service -n churn-prediction
echo.
echo Or get URLs only:
echo   minikube service api-service -n churn-prediction --url
echo   minikube service streamlit-service -n churn-prediction --url
echo.
echo Useful commands:
echo   kubectl get pods -n churn-prediction
echo   kubectl logs -f deployment/api -n churn-prediction
echo   minikube dashboard
echo.

pause
