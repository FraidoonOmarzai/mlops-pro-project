#!/bin/bash

# Simple Cleanup Script for AWS Resources
# Usage: ./scripts/cleanup-simple.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CLUSTER_NAME="churn-prediction-cluster"

echo -e "${YELLOW}=================================${NC}"
echo -e "${YELLOW}AWS Cleanup${NC}"
echo -e "${YELLOW}=================================${NC}"
echo ""
echo -e "${RED}⚠️  This will delete all AWS resources!${NC}"
echo ""
read -p "Are you sure? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

# Delete Kubernetes resources
echo -e "${YELLOW}Deleting Kubernetes resources...${NC}"
kubectl delete namespace churn-prediction 2>/dev/null || echo "Namespace already deleted"

echo -e "${GREEN}✓ Kubernetes resources deleted${NC}"

# Instructions for deleting EKS cluster
echo ""
echo -e "${YELLOW}To delete the EKS cluster:${NC}"
echo ""
echo "Option 1 - AWS Console:"
echo "  1. Go to: https://console.aws.amazon.com/eks"
echo "  2. Select: ${CLUSTER_NAME}"
echo "  3. Delete node groups first"
echo "  4. Then delete cluster"
echo ""
echo "Option 2 - AWS CLI:"
echo "  aws eks delete-cluster --name ${CLUSTER_NAME} --region us-east-1"
echo ""

echo -e "${GREEN}✓ Cleanup complete${NC}"
