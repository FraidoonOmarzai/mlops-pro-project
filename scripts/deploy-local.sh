#!/bin/bash

# Deploy to Local Kubernetes (Minikube)
# Usage: ./scripts/deploy-local.sh YOUR_DOCKERHUB_USERNAME

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo "Usage: ./scripts/deploy-local.sh YOUR_DOCKERHUB_USERNAME"
    exit 1
fi

DOCKER_USERNAME=$1

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Deploying to Local Kubernetes${NC}"
echo -e "${GREEN}=================================${NC}"

# Check if minikube is running
if ! minikube status | grep -q "Running"; then
    echo -e "${YELLOW}Starting minikube...${NC}"
    minikube start
fi

# Create namespace
echo -e "${YELLOW}Creating namespace...${NC}"
kubectl apply -f k8s/namespace.yaml

# Update YAML files with Docker username
echo -e "${YELLOW}Updating YAML files...${NC}"
mkdir -p k8s/temp
sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/api.yaml > k8s/temp/api.yaml
sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/streamlit.yaml > k8s/temp/streamlit.yaml

# Deploy API
echo -e "${YELLOW}Deploying API...${NC}"
kubectl apply -f k8s/temp/api.yaml

# Deploy Streamlit
echo -e "${YELLOW}Deploying Streamlit...${NC}"
kubectl apply -f k8s/temp/streamlit.yaml

# Wait for pods
echo -e "${YELLOW}Waiting for pods to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/api -n churn-prediction || true
kubectl wait --for=condition=available --timeout=300s deployment/streamlit -n churn-prediction || true

# Get service URLs
echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""
echo -e "${GREEN}Access your services:${NC}"
echo ""

# Get API URL
echo "Getting API URL..."
minikube service api-service -n churn-prediction --url &

# Get Streamlit URL
echo "Getting Streamlit URL..."
minikube service streamlit-service -n churn-prediction --url &

sleep 3

echo ""
echo -e "${YELLOW}Or run these commands:${NC}"
echo "  minikube service api-service -n churn-prediction"
echo "  minikube service streamlit-service -n churn-prediction"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  kubectl get pods -n churn-prediction"
echo "  kubectl logs -f deployment/api -n churn-prediction"
echo "  minikube dashboard"
echo ""
