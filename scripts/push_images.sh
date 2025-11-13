#!/bin/bash

# Script to push Docker images to Docker Hub
# Usage: ./scripts/push_images.sh [version]

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
    exit 1
fi

# Get version
VERSION=${1:-${VERSION:-latest}}

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Pushing Docker Images to Docker Hub${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "Version: ${YELLOW}${VERSION}${NC}"
echo -e "Docker Username: ${YELLOW}${DOCKER_USERNAME}${NC}"
echo ""

# Check if logged in to Docker Hub
if ! docker info | grep Username > /dev/null 2>&1; then
    echo -e "${YELLOW}Not logged in to Docker Hub${NC}"
    echo -e "Logging in..."
    docker login
fi

# Push API image
echo -e "${GREEN}[1/3] Pushing API image...${NC}"
docker push ${DOCKER_USERNAME}/churn-prediction-api:${VERSION}
docker push ${DOCKER_USERNAME}/churn-prediction-api:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ API image pushed successfully${NC}"
else
    echo -e "${RED}✗ Failed to push API image${NC}"
    exit 1
fi

# Push Streamlit image
echo -e "${GREEN}[2/3] Pushing Streamlit image...${NC}"
docker push ${DOCKER_USERNAME}/churn-prediction-streamlit:${VERSION}
docker push ${DOCKER_USERNAME}/churn-prediction-streamlit:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Streamlit image pushed successfully${NC}"
else
    echo -e "${RED}✗ Failed to push Streamlit image${NC}"
    exit 1
fi

# Push Training image
echo -e "${GREEN}[3/3] Pushing Training image...${NC}"
docker push ${DOCKER_USERNAME}/churn-prediction-training:${VERSION}
docker push ${DOCKER_USERNAME}/churn-prediction-training:latest

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training image pushed successfully${NC}"
else
    echo -e "${RED}✗ Failed to push Training image${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Push Summary${NC}"
echo -e "${GREEN}======================================${NC}"
echo "Images pushed to Docker Hub:"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-api:${VERSION}"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-streamlit:${VERSION}"
echo -e "  • ${DOCKER_USERNAME}/churn-prediction-training:${VERSION}"
echo ""
echo "View your images at:"
echo -e "  ${YELLOW}https://hub.docker.com/u/${DOCKER_USERNAME}${NC}"
echo ""
echo -e "${GREEN}Images are now available for deployment!${NC}"
echo ""