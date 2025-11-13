#!/bin/bash

# Script to build Docker images for all services
# Usage: ./scripts/build_images.sh [version]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Get version from argument or use latest
VERSION=${1:-${VERSION:-latest}}

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Building Docker Images${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Version: ${YELLOW}${VERSION}${NC}"
echo -e "Docker Username: ${YELLOW}${DOCKER_USERNAME}${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build API image
echo -e "${GREEN}[1/3] Building API image...${NC}"
docker build \
    -t ${DOCKER_USERNAME}/churn-prediction-api:${VERSION} \
    -t ${DOCKER_USERNAME}/churn-prediction-api:latest \
    -f docker/Dockerfile.api \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ API image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build API image${NC}"
    exit 1
fi

# Build Streamlit image
echo -e "${GREEN}[2/3] Building Streamlit image...${NC}"
docker build \
    -t ${DOCKER_USERNAME}/churn-prediction-streamlit:${VERSION} \
    -t ${DOCKER_USERNAME}/churn-prediction-streamlit:latest \
    -f docker/Dockerfile.streamlit \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Streamlit image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Streamlit image${NC}"
    exit 1
fi

# Build Training image
echo -e "${GREEN}[3/3] Building Training image...${NC}"
docker build \
    -t ${DOCKER_USERNAME}/churn-prediction-training:${VERSION} \
    -t ${DOCKER_USERNAME}/churn-prediction-training:latest \
    -f docker/Dockerfile.training \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Training image${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Images built:"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-api:${VERSION}"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-streamlit:${VERSION}"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-training:${VERSION}"
echo ""
echo "List all images:"
docker images | grep churn-prediction

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Test images: docker-compose up"
echo "  2. Push to Docker Hub: ./scripts/push_images.sh"
echo ""