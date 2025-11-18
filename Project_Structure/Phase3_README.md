# Phase 3: Docker Containerization - Complete Guide

## 🐳 Overview

Phase 3 containerizes the entire application using Docker, enabling consistent deployments across any environment.

---

## 📦 What's Been Created

### **Docker Images (3)**
1. **API Image** - FastAPI service
2. **Streamlit Image** - Web dashboard
3. **Training Image** - Model training pipeline

### **Configuration Files (7)**
1. `docker/Dockerfile.api` - API container definition
2. `docker/Dockerfile.streamlit` - Streamlit container
3. `docker/Dockerfile.training` - Training container
4. `docker-compose.yml` - Multi-container orchestration
5. `.dockerignore` - Exclude unnecessary files
6. `.env.example` - Environment template
7. Build & push scripts

---

```bash

mlops-churn-prediction/
│
├── docker/
│   ├── Dockerfile.api           # NEW - API container
│   ├── Dockerfile.streamlit     # NEW - Streamlit container
│   └── Dockerfile.training      # NEW - Training container
│
├── docker-compose.yml           # NEW - Orchestration
├── .dockerignore                # NEW - Exclude files
├── .env.example                 # NEW - Environment template
│
└── scripts/
    ├── build_images.sh          # NEW - Build script
    ├── push_images.sh           # NEW - Push to Docker Hub
    └── run_docker.sh            # NEW - Run containers
```

## 🚀 Quick Start

### **Step 1: Prerequisites**
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

### **Step 2: Setup Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file
# Set DOCKER_USERNAME to your Docker Hub username
nano .env  # or use your favorite editor
```

Example `.env`:
```bash
DOCKER_USERNAME=yourusername
VERSION=v1.0.0
```

### **Step 3: Make Scripts Executable (Linux/Mac)**
```bash
chmod +x scripts/build_images.sh
chmod +x scripts/push_images.sh
chmod +x scripts/run_docker.sh
```

### **Step 4: Build Images**

**Linux/Mac:**
```bash
./scripts/build_images.sh
```

**Windows:**
```cmd
scripts\build_images.bat
```

This will build all 3 Docker images (~5-10 minutes first time).

### **Step 5: Run with Docker Compose**
```bash
# Start all services
docker-compose up -d

# Or use the helper script
./scripts/run_docker.sh start
```

### **Step 6: Access Services**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501
- **MLflow**: http://localhost:5000

---

## 🏗️ Docker Architecture

```
┌─────────────────────────────────────────┐
│         Docker Compose Network          │
│                                         │
│  ┌──────────┐  ┌──────────┐   ┌──────┐  │
│  │   API    │  │Streamlit │   │MLflow│  │
│  │  :8000   │  │  :8501   │   │ :5000│  │
│  └────┬─────┘  └────┬─────┘   └───┬──┘  │
│       │             │             │     │
│       └─────────────┴─────────────┘     │
│            Shared Network               │
└─────────────────────────────────────────┘
         │                    │
    [Volumes]            [Volumes]
    artifacts/            mlflow/
     logs/                 data/
```

---

## 📝 Detailed Commands

### **Building Images**

```bash
# Build all images
./scripts/build_images.sh

# Build with specific version
./scripts/build_images.sh v1.0.0

# Build individual image
docker build -t username/churn-prediction-api:latest -f docker/Dockerfile.api .
```

### **Running Containers**

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### **Managing Services**

```bash
# Check status
docker-compose ps

# Restart service
docker-compose restart api

# Scale service (if needed)
docker-compose up -d --scale api=3

# Execute command in container
docker-compose exec api bash
```

### **Using Helper Script**

```bash
# Start services
./scripts/run_docker.sh start

# Stop services
./scripts/run_docker.sh stop

# Restart services
./scripts/run_docker.sh restart

# View logs
./scripts/run_docker.sh logs

# View status
./scripts/run_docker.sh status

# Run training
./scripts/run_docker.sh train

# Cleanup
./scripts/run_docker.sh cleanup
```

---

## 🚢 Pushing to Docker Hub

### **Step 1: Create Docker Hub Account**
1. Go to https://hub.docker.com
2. Sign up for free account
3. Create access token (Settings → Security → New Access Token)

### **Step 2: Login to Docker Hub**
```bash
docker login
# Enter username and password/token
```

### **Step 3: Push Images**

**Linux/Mac:**
```bash
./scripts/push_images.sh
```

**Windows:**
```cmd
scripts\push_images.bat
```

**Manual Push:**
```bash
docker push username/churn-prediction-api:latest
docker push username/churn-prediction-streamlit:latest
docker push username/churn-prediction-training:latest
```

### **Step 4: Verify**
Visit: https://hub.docker.com/u/yourusername

---

## 🔍 Image Details

### **API Image**
- **Base**: Python 3.10-slim
- **Size**: ~500MB
- **Exposed Port**: 8000
- **Health Check**: Every 30s
- **Features**:
  - Multi-stage build (optimized size)
  - FastAPI + Uvicorn
  - Auto-reload disabled for production

### **Streamlit Image**
- **Base**: Python 3.10-slim
- **Size**: ~550MB
- **Exposed Port**: 8501
- **Health Check**: Every 30s
- **Features**:
  - Streamlit server configuration
  - Interactive dashboard
  - Plotly visualizations

### **Training Image**
- **Base**: Python 3.10-slim
- **Size**: ~500MB
- **Features**:
  - Complete training pipeline
  - MLflow integration
  - Volume mounts for artifacts

### **MLflow Image**
- **Base**: Official MLflow image
- **Size**: ~800MB
- **Exposed Port**: 5000
- **Features**:
  - SQLite backend
  - Artifact storage
  - Experiment tracking

---

## 📊 Docker Compose Services

### **Service: api**
- **Image**: churn-prediction-api
- **Port**: 8000
- **Dependencies**: mlflow
- **Volumes**:
  - `./artifacts:/app/artifacts:ro` (read-only)
  - `./logs:/app/logs`
  - `./config:/app/config:ro`
- **Restart Policy**: unless-stopped

### **Service: streamlit**
- **Image**: churn-prediction-streamlit
- **Port**: 8501
- **Dependencies**: api
- **Volumes**: Same as API
- **Environment**: API_URL=http://api:8000

### **Service: mlflow**
- **Image**: Official MLflow
- **Port**: 5000
- **Volume**: mlflow-data (persistent)
- **Backend**: SQLite

### **Service: training**
- **Image**: churn-prediction-training
- **Profile**: training (run on-demand)
- **Volumes**: All project directories
- **Usage**: `docker-compose --profile training up training`

---

## 🔧 Customization

### **Change Ports**

Edit `docker-compose.yml`:
```yaml
services:
  api:
    ports:
      - "8080:8000"  # Change 8080 to your port
```

### **Add Environment Variables**

```yaml
services:
  api:
    environment:
      - CUSTOM_VAR=value
      - LOG_LEVEL=DEBUG
```

### **Use Different Model**

Edit `.env`:
```bash
MODEL_PATH=artifacts/models/random_forest.pkl
```

### **Memory Limits**

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

---

## 🧪 Testing Dockerized Services

### **Test API Health**
```bash
curl http://localhost:8000/health
```

### **Test Prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### **Test Streamlit**
Open browser: http://localhost:8501

### **Test MLflow**
Open browser: http://localhost:5000

### **Run Training in Container**
```bash
docker-compose --profile training up training
```

---

## 🐛 Troubleshooting

### **Issue: Port already in use**
```bash
# Find process using port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Change port in docker-compose.yml or stop other service
```

### **Issue: Image build fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### **Issue: Container won't start**
```bash
# Check logs
docker-compose logs api

# Check container status
docker ps -a

# Restart service
docker-compose restart api
```

### **Issue: Cannot connect to services**
```bash
# Check network
docker network ls
docker network inspect churn-prediction-network

# Verify containers are in same network
docker inspect <container_name> | grep NetworkMode
```

### **Issue: Volume mount permission denied**
```bash
# On Linux, check ownership
ls -la artifacts/

# Fix permissions
sudo chown -R $USER:$USER artifacts/
```

### **Issue: Out of disk space**
```bash
# Clean unused Docker resources
docker system prune -a --volumes

# Check disk usage
docker system df
```

---

## 📈 Performance Optimization

### **Multi-stage Builds**
Already implemented in Dockerfiles to reduce image size.

### **Layer Caching**
Order Dockerfile commands from least to most frequently changing.

### **Use .dockerignore**
Already configured to exclude unnecessary files.

### **Minimize Layers**
Combine RUN commands where possible.

### **Use Specific Tags**
Always use specific versions, not just `latest`.

---

## 🔒 Security Best Practices

### **1. Don't Run as Root**
Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### **2. Scan Images**
```bash
docker scan username/churn-prediction-api:latest
```

### **3. Use Official Base Images**
Already using official Python images.

### **4. Keep Images Updated**
```bash
# Rebuild regularly
docker-compose build --pull
```

### **5. Secure Secrets**
Use Docker secrets or environment variables, never hardcode.

---

## 📊 Monitoring

### **Container Stats**
```bash
# Real-time stats
docker stats

# Specific container
docker stats churn-prediction-api
```

### **Logs**
```bash
# All logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100
```

### **Health Checks**
```bash
# Check health
docker ps

# Inspect health
docker inspect churn-prediction-api | grep Health -A 10
```

---

## 🎯 Production Deployment

### **Using Built Images**

On any machine with Docker:
```bash
# Pull images
docker pull username/churn-prediction-api:latest
docker pull username/churn-prediction-streamlit:latest

# Run
docker-compose up -d
```

### **Environment-Specific Configs**

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

---

## 📝 Next Steps (Phase 4)

Once Docker is working:
1. ✅ Write comprehensive tests
2. ✅ Unit tests for all components
3. ✅ Integration tests for APIs
4. ✅ Data validation tests
5. ✅ Model performance tests
6. ✅ Test coverage reporting

---

## ✅ Phase 3 Checklist

- ✅ Dockerfiles created for all services
- ✅ Docker Compose configuration complete
- ✅ Multi-stage builds for optimization
- ✅ Health checks implemented
- ✅ Build scripts created
- ✅ Push scripts created
- ✅ Volume mounts configured
- ✅ Network configuration set
- ✅ .dockerignore configured
- ✅ Environment variables templated

---

## 🎉 Success Criteria

Phase 3 is complete when:
1. ✅ All images build successfully
2. ✅ Docker Compose starts all services
3. ✅ API is accessible at port 8000
4. ✅ Streamlit is accessible at port 8501
5. ✅ MLflow is accessible at port 5000
6. ✅ Images pushed to Docker Hub
7. ✅ Services communicate correctly
8. ✅ Health checks pass

---

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com
- **Docker Compose**: https://docs.docker.com/compose
- **Docker Hub**: https://hub.docker.com
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices

---

**Phase 3 Complete! Ready for Phase 4: Testing?** 🚀

Test your Docker setup and let me know when you're ready to proceed!
