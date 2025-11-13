#!/bin/bash

# Script to run Docker containers using docker-compose
# Usage: ./scripts/run_docker.sh [start|stop|restart|logs|status]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Commands
COMMAND=${1:-start}

# Function to start services
start_services() {
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Starting Churn Prediction Services${NC}"
    echo -e "${GREEN}======================================${NC}"
    
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}Services started successfully!${NC}"
    echo ""
    echo "Access the services:"
    echo -e "  • API:        ${YELLOW}http://localhost:8000${NC}"
    echo -e "  • API Docs:   ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "  • Streamlit:  ${YELLOW}http://localhost:8501${NC}"
    echo -e "  • MLflow:     ${YELLOW}http://localhost:5000${NC}"
    echo ""
    echo "View logs: ./scripts/run_docker.sh logs"
    echo "Stop services: ./scripts/run_docker.sh stop"
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}Services stopped${NC}"
}

# Function to restart services
restart_services() {
    echo -e "${YELLOW}Restarting services...${NC}"
    docker-compose restart
    echo -e "${GREEN}Services restarted${NC}"
}

# Function to show logs
show_logs() {
    SERVICE=${2:-}
    if [ -z "$SERVICE" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f $SERVICE
    fi
}

# Function to show status
show_status() {
    echo -e "${GREEN}Service Status:${NC}"
    docker-compose ps
    echo ""
    echo -e "${GREEN}Container Health:${NC}"
    docker ps --filter "name=churn-prediction" --format "table {{.Names}}\t{{.Status}}"
}

# Function to run training
run_training() {
    echo -e "${GREEN}Starting model training in container...${NC}"
    docker-compose --profile training up training
}

# Function to clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up containers and volumes...${NC}"
    docker-compose down -v
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Main script
case $COMMAND in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs $@
        ;;
    status)
        show_status
        ;;
    train)
        run_training
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|train|cleanup}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - View logs (optional: specify service name)"
        echo "  status   - Show service status"
        echo "  train    - Run training pipeline"
        echo "  cleanup  - Stop and remove containers and volumes"
        exit 1
        ;;
esac