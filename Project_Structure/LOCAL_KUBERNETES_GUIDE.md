# Running Kubernetes Locally - Complete Guide

## 🎯 Choose Your Method

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Minikube** | Easy, full K8s features | Uses VM/Docker | Learning K8s |
| **Docker Desktop** | Integrated, simple | Windows/Mac only | Docker users |
| **Kind** | Fast, lightweight | Requires Docker | CI/CD testing |
| **K3s** | Very lightweight | Linux best | Production-like |

---

## ⭐ METHOD 1: Minikube (Recommended)

### Installation

**Mac:**
```bash
brew install minikube kubectl
```

**Windows (PowerShell as Admin):**
```powershell
choco install minikube kubernetes-cli
```

**Linux:**
```bash
# Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/kubectl
```

### Quick Start

```bash
# Start minikube
minikube start

# Check status
minikube status

# Should show:
# minikube
# type: Control Plane
# host: Running
# kubelet: Running
# apiserver: Running
# kubeconfig: Configured
```

### Deploy Your App

```bash
# Make deploy script executable
chmod +x scripts/deploy-local.sh

# Deploy (replace with your Docker Hub username)
./scripts/deploy-local.sh YOUR_DOCKERHUB_USERNAME

# Example:
./scripts/deploy-local.sh johndoe
```

### Access Services

```bash
# Method 1: Auto-open in browser
minikube service api-service -n churn-prediction
minikube service streamlit-service -n churn-prediction

# Method 2: Get URLs
minikube service api-service -n churn-prediction --url
minikube service streamlit-service -n churn-prediction --url

# Example output:
# http://192.168.49.2:30000  (API)
# http://192.168.49.2:30001  (Streamlit)
```

### Useful Commands

```bash
# Check pods
kubectl get pods -n churn-prediction

# View logs
kubectl logs -f deployment/api -n churn-prediction

# Open dashboard
minikube dashboard

# Stop minikube (saves state)
minikube stop

# Delete minikube (removes everything)
minikube delete

# SSH into minikube
minikube ssh
```

---

## ⭐ METHOD 2: Docker Desktop Kubernetes

### Enable Kubernetes

1. Open **Docker Desktop**
2. Click **Settings** (⚙️)
3. Go to **Kubernetes** tab
4. Check ☑️ **Enable Kubernetes**
5. Click **Apply & Restart**
6. Wait 2-3 minutes for K8s to start

### Verify

```bash
# Check context
kubectl config current-context
# Should show: docker-desktop

# Check nodes
kubectl get nodes
# Should show: docker-desktop   Ready
```

### Deploy Your App

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy API (replace YOUR_DOCKERHUB_USERNAME)
sed 's/YOUR_DOCKERHUB_USERNAME/yourusername/g' k8s/api.yaml | kubectl apply -f -

# Deploy Streamlit
sed 's/YOUR_DOCKERHUB_USERNAME/yourusername/g' k8s/streamlit.yaml | kubectl apply -f -

# Check status
kubectl get pods -n churn-prediction
kubectl get svc -n churn-prediction
```

### Access Services

Since Docker Desktop doesn't have `minikube service`, use port-forwarding:

```bash
# Terminal 1: Forward API
kubectl port-forward -n churn-prediction svc/api-service 8000:80

# Terminal 2: Forward Streamlit
kubectl port-forward -n churn-prediction svc/streamlit-service 8501:80
```

**Access:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501

---

## 🚀 METHOD 3: Kind (Kubernetes in Docker)

### Installation

```bash
# Mac
brew install kind

# Windows
choco install kind

# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

### Create Cluster

```bash
# Create cluster with port mappings
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30000
    hostPort: 8000
  - containerPort: 30001
    hostPort: 8501
EOF

# Verify
kubectl cluster-info
```

### Deploy Your App

```bash
# Same as Minikube
kubectl apply -f k8s/namespace.yaml
# ... deploy API and Streamlit
```

### Access Services

Use NodePort with localhost:
- API: http://localhost:8000
- Streamlit: http://localhost:8501

### Delete Cluster

```bash
kind delete cluster
```

---

## 📊 Comparison Table

| Feature | Minikube | Docker Desktop | Kind |
|---------|----------|----------------|------|
| **Easy Setup** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Resource Usage** | Medium | Low | Low |
| **Speed** | Medium | Fast | Fast |
| **Dashboard** | ✅ Built-in | ✅ Built-in | ❌ Manual |
| **LoadBalancer** | ✅ Yes | ❌ No | ❌ No |
| **Multi-node** | ✅ Yes | ❌ No | ✅ Yes |
| **Platform** | All | Win/Mac | All |

---

## 🔧 Common Issues & Solutions

### Issue: Minikube won't start

```bash
# Check if VirtualBox/Docker is running
minikube start --driver=docker

# Or try specific driver
minikube start --driver=virtualbox

# Delete and recreate
minikube delete
minikube start
```

### Issue: kubectl not connecting

```bash
# Set correct context
kubectl config use-context minikube

# Or for Docker Desktop
kubectl config use-context docker-desktop

# View all contexts
kubectl config get-contexts
```

### Issue: Pods not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n churn-prediction

# Common issues:
# - ImagePullBackOff: Image doesn't exist or is private
# - CrashLoopBackOff: App crashes on startup

# View logs
kubectl logs <pod-name> -n churn-prediction
```

### Issue: Can't access services

**For Minikube:**
```bash
# Get service URL
minikube service <service-name> -n <namespace> --url
```

**For Docker Desktop:**
```bash
# Use port-forward
kubectl port-forward -n churn-prediction svc/api-service 8000:80
```

### Issue: Out of resources

```bash
# Give minikube more resources
minikube start --memory=4096 --cpus=2

# Or delete and recreate
minikube delete
minikube start --memory=4096 --cpus=2
```

---

## 💡 Best Practices for Local Development

### 1. Use Lower Resource Limits

```yaml
resources:
  requests:
    memory: "256Mi"  # Lower for local
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "250m"
```

### 2. Use Fewer Replicas

```yaml
spec:
  replicas: 1  # Just 1 for local testing
```

### 3. Use NodePort Instead of LoadBalancer

```yaml
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8000
    nodePort: 30000  # Fixed port
```

### 4. Enable Docker Registry Cache (Minikube)

```bash
minikube addons enable registry
```

### 5. Use Local Images (Skip Docker Hub)

```bash
# Build directly in minikube
eval $(minikube docker-env)
docker build -t churn-api:local -f docker/Dockerfile.api .

# Use in deployment
image: churn-api:local
imagePullPolicy: Never
```

---

## 📚 Essential Commands Cheat Sheet

### Minikube

```bash
minikube start              # Start cluster
minikube stop               # Stop cluster
minikube delete             # Delete cluster
minikube status             # Check status
minikube service <name>     # Access service
minikube dashboard          # Open dashboard
minikube ip                 # Get cluster IP
minikube ssh                # SSH into node
minikube logs               # View logs
```

### kubectl

```bash
# Pods
kubectl get pods -n <namespace>
kubectl describe pod <pod> -n <namespace>
kubectl logs -f <pod> -n <namespace>
kubectl delete pod <pod> -n <namespace>

# Deployments
kubectl get deployments -n <namespace>
kubectl scale deployment <name> --replicas=2
kubectl rollout restart deployment/<name>

# Services
kubectl get svc -n <namespace>
kubectl port-forward svc/<name> 8000:80

# Namespaces
kubectl get namespaces
kubectl create namespace <name>
kubectl delete namespace <name>

# Everything
kubectl get all -n <namespace>
kubectl delete all --all -n <namespace>
```

---

## 🎯 Quick Start Script

Save this as `quick-start-local.sh`:

```bash
#!/bin/bash

# Quick start local Kubernetes
echo "Starting Minikube..."
minikube start

echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo "Enter your Docker Hub username:"
read DOCKER_USERNAME

echo "Deploying services..."
sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/api.yaml | kubectl apply -f -
sed "s/YOUR_DOCKERHUB_USERNAME/${DOCKER_USERNAME}/g" k8s/streamlit.yaml | kubectl apply -f -

echo "Waiting for pods..."
kubectl wait --for=condition=available --timeout=300s deployment/api -n churn-prediction
kubectl wait --for=condition=available --timeout=300s deployment/streamlit -n churn-prediction

echo "Opening services..."
minikube service api-service -n churn-prediction &
minikube service streamlit-service -n churn-prediction &

echo "Done! Check the opened browser tabs."
```

Usage:
```bash
chmod +x quick-start-local.sh
./quick-start-local.sh
```

---

## ✅ Local K8s Checklist

- [ ] Chose a method (Minikube/Docker Desktop/Kind)
- [ ] Installed kubectl
- [ ] Installed local K8s (minikube/etc)
- [ ] Started cluster
- [ ] Verified with `kubectl get nodes`
- [ ] Created namespace
- [ ] Deployed API and Streamlit
- [ ] Pods are Running
- [ ] Can access services
- [ ] Tested making predictions

---

## 🎓 Learning Path

1. **Start with Minikube** - Easiest and most features
2. **Learn kubectl commands** - Get, describe, logs
3. **Understand Deployments** - How apps run
4. **Learn Services** - How to access apps
5. **Try Docker Desktop** - If you prefer simplicity
6. **Experiment with Kind** - For advanced use cases

---

## 💰 Resource Usage

**Minikube:**
- RAM: 2-4 GB
- CPU: 2 cores
- Disk: 20 GB

**Docker Desktop K8s:**
- RAM: 2-3 GB (shared with Docker)
- CPU: 2 cores
- Disk: 10 GB

**Kind:**
- RAM: 1-2 GB
- CPU: 1-2 cores
- Disk: 5 GB

---

## 🎉 Success!

When you see your pods running and can access the services, you've successfully run Kubernetes locally!

**Benefits:**
- ✅ Test before deploying to cloud
- ✅ Free (no AWS costs)
- ✅ Fast development cycle
- ✅ Learn Kubernetes
- ✅ Debug issues easily

---

## ❓ FAQ

**Q: Which method should I use?**
A: Start with Minikube - it's the most feature-complete and easiest to learn.

**Q: Do I need to push images to Docker Hub?**
A: Yes, unless you build images inside minikube using `eval $(minikube docker-env)`.

**Q: Can I use this for production?**
A: No, local K8s is for development/testing only. Use AWS/GCP/Azure for production.

**Q: How do I stop eating my laptop battery?**
A: Run `minikube stop` when not using it. It saves the state and you can `minikube start` later.

**Q: Why is it slow?**
A: Give it more resources: `minikube start --memory=4096 --cpus=2`

---

**That's it! You're now running Kubernetes locally!** 🎉

Start with Minikube, it's the easiest and most powerful option for learning.
