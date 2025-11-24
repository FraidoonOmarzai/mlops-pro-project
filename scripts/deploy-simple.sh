#!/bin/bash

# Simple AWS EKS Deployment Script
# Usage: ./scripts/deploy-simple.sh YOUR_DOCKERHUB_USERNAME

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Docker username is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Docker Hub username required${NC}"
    echo "Usage: ./scripts/deploy-simple.sh YOUR_DOCKERHUB_USERNAME"
    exit 1
fi

DOCKER_USERNAME=$1
CLUSTER_NAME="churn-prediction-cluster"

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Simple AWS EKS Deployment${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI not found. Install from: https://aws.amazon.com/cli/${NC}"
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/${NC}"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured. Run: aws configure${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ All prerequisites met${NC}"
}

# Function to create EKS cluster (if not exists)
create_cluster() {
    echo -e "${YELLOW}Checking for EKS cluster...${NC}"

    if aws eks describe-cluster --name ${CLUSTER_NAME} --region us-east-1 &> /dev/null; then
        echo -e "${GREEN}✓ Cluster already exists${NC}"
    else
        echo -e "${YELLOW}Creating EKS cluster (this takes ~15 minutes)...${NC}"
        echo -e "${YELLOW}You can also create it manually in AWS Console${NC}"
        echo ""
        echo "Go to: https://console.aws.amazon.com/eks"
        echo "Click: Add cluster → Create"
        echo "Cluster name: ${CLUSTER_NAME}"
        echo "Kubernetes version: 1.28"
        echo "Node group: t3.medium, 2 nodes"
        echo ""
        read -p "Press Enter after creating the cluster..."
    fi

    # Update kubeconfig
    echo -e "${YELLOW}Updating kubeconfig...${NC}"
    aws eks update-kubeconfig --name ${CLUSTER_NAME} --region us-east-1

    echo -e "${GREEN}✓ Connected to cluster${NC}"
}

# Function to update Docker username in YAML files
update_yaml_files() {
    echo -e "${YELLOW}Updating YAML files with your Docker username...${NC}"

    # Create temp directory
    mkdir -p k8s/temp

    # Update API YAML
    sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/api.yaml > k8s/temp/api.yaml

    # Update Streamlit YAML
    sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/streamlit.yaml > k8s/temp/streamlit.yaml

    echo -e "${GREEN}✓ YAML files updated${NC}"
}

# Function to deploy to Kubernetes
deploy_to_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"

    # Create namespace
    kubectl apply -f k8s/namespace.yaml

    # Deploy API
    kubectl apply -f k8s/temp/api.yaml

    # Deploy Streamlit
    kubectl apply -f k8s/temp/streamlit.yaml

    echo -e "${GREEN}✓ Deployed to Kubernetes${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "${YELLOW}Waiting for pods to be ready (this may take 2-3 minutes)...${NC}"

    kubectl wait --for=condition=available --timeout=300s deployment/api -n churn-prediction || true
    kubectl wait --for=condition=available --timeout=300s deployment/streamlit -n churn-prediction || true

    echo -e "${GREEN}✓ Deployments ready${NC}"
}

# Function to get service URLs
get_urls() {
    echo -e "${YELLOW}Getting service URLs...${NC}"
    echo ""

    echo "Waiting for LoadBalancers (this may take 2-3 minutes)..."
    sleep 30

    API_URL=$(kubectl get svc api-service -n churn-prediction -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")
    STREAMLIT_URL=$(kubectl get svc streamlit-service -n churn-prediction -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")

    echo ""
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo ""
    echo -e "${GREEN}Your Services:${NC}"
    echo ""
    if [ "$API_URL" != "pending" ]; then
        echo -e "  API URL:        ${YELLOW}http://${API_URL}${NC}"
        echo -e "  API Docs:       ${YELLOW}http://${API_URL}/docs${NC}"
    else
        echo -e "  API URL:        ${YELLOW}Still provisioning...${NC}"
    fi

    if [ "$STREAMLIT_URL" != "pending" ]; then
        echo -e "  Streamlit URL:  ${YELLOW}http://${STREAMLIT_URL}${NC}"
    else
        echo -e "  Streamlit URL:  ${YELLOW}Still provisioning...${NC}"
    fi
    echo ""

    if [ "$API_URL" = "pending" ] || [ "$STREAMLIT_URL" = "pending" ]; then
        echo -e "${YELLOW}Note: URLs are still being provisioned. Run this to check:${NC}"
        echo "  kubectl get svc -n churn-prediction"
    fi
}

# Function to show helpful commands
show_commands() {
    echo ""
    echo -e "${GREEN}Useful Commands:${NC}"
    echo ""
    echo "  # Check status"
    echo "  kubectl get pods -n churn-prediction"
    echo "  kubectl get svc -n churn-prediction"
    echo ""
    echo "  # View logs"
    echo "  kubectl logs -f deployment/api -n churn-prediction"
    echo "  kubectl logs -f deployment/streamlit -n churn-prediction"
    echo ""
    echo "  # Scale deployment"
    echo "  kubectl scale deployment api --replicas=3 -n churn-prediction"
    echo ""
    echo "  # Delete everything"
    echo "  kubectl delete namespace churn-prediction"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    create_cluster
    update_yaml_files
    deploy_to_kubernetes
    wait_for_deployment
    get_urls
    show_commands
}

# Run main function
main
