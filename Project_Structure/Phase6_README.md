# Phase 6: AWS Deployment - Simple & Easy Guide

## 🎯 What This Does

Deploys your app to AWS so anyone can access it on the internet!

---

## 📦 Files Created (6 files)

1. `k8s/namespace.yaml` - Kubernetes namespace
2. `k8s/api.yaml` - API deployment
3. `k8s/streamlit.yaml` - Streamlit deployment
4. `scripts/deploy-simple.sh` - One-command deployment
5. `scripts/cleanup-simple.sh` - Easy cleanup
6. This README

**Total: ~250 lines** vs 1000+ in complex version!

---

## 🚀 Quick Start (30 minutes)

### Step 1: Install Tools (5 min)

**Install AWS CLI:**
```bash
# Mac
brew install awscli

# Windows
# Download from: https://aws.amazon.com/cli/

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Install kubectl:**
```bash
# Mac
brew install kubectl

# Windows
# Download from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

**Verify:**
```bash
aws --version
kubectl version --client
```

### Step 2: Configure AWS (2 min)

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: [Your AWS key]
- **AWS Secret Access Key**: [Your AWS secret]
- **Default region**: `us-east-1`
- **Default output format**: `json`

**Get AWS keys from:**
1. Go to: https://console.aws.amazon.com/iam/
2. Click: Users → Your user → Security credentials
3. Create access key

### Step 3: Create EKS Cluster (15 min)

**Option A: AWS Console (Easier)**

1. Go to: https://console.aws.amazon.com/eks
2. Click: **Add cluster** → **Create**
3. Settings:
   - **Name**: `churn-prediction-cluster`
   - **Kubernetes version**: `1.28`
   - **Cluster service role**: Create new (if needed)
   - **VPC**: Default VPC
   - Click: **Next** through defaults
4. Click: **Create**
5. Wait ~10 minutes
6. After cluster is ready, add **Node Group**:
   - Name: `churn-nodes`
   - Instance type: `t3.medium`
   - Nodes: `2`
   - Click: **Create**

**Option B: AWS CLI (Faster if you know what you're doing)**

```bash
# Install eksctl
brew install eksctl  # Mac
# Or download from: https://eksctl.io/

# Create cluster
eksctl create cluster \
  --name churn-prediction-cluster \
  --region us-east-1 \
  --nodes 2 \
  --node-type t3.medium
```

### Step 4: Deploy Your App (5 min)

```bash
# Make script executable
chmod +x scripts/deploy-simple.sh

# Deploy (replace with YOUR Docker Hub username)
./scripts/deploy-simple.sh YOUR_DOCKERHUB_USERNAME

# Example:
./scripts/deploy-simple.sh johndoe
```

**That's it!** The script does everything:
- ✅ Checks prerequisites
- ✅ Connects to cluster
- ✅ Deploys API and Streamlit
- ✅ Creates LoadBalancers
- ✅ Shows your URLs

### Step 5: Access Your App (2 min)

Wait 2-3 minutes for LoadBalancers, then:

```bash
# Get URLs
kubectl get svc -n churn-prediction

# You'll see something like:
# api-service        LoadBalancer   xxx.us-east-1.elb.amazonaws.com
# streamlit-service  LoadBalancer   yyy.us-east-1.elb.amazonaws.com
```

**Access:**
- API: `http://xxx.us-east-1.elb.amazonaws.com`
- API Docs: `http://xxx.us-east-1.elb.amazonaws.com/docs`
- Streamlit: `http://yyy.us-east-1.elb.amazonaws.com`

---

## 📊 Check Status

```bash
# Check if pods are running
kubectl get pods -n churn-prediction

# Should show:
# NAME                         READY   STATUS    RESTARTS   AGE
# api-xxxxxxx-xxxxx           1/1     Running   0          2m
# api-xxxxxxx-xxxxx           1/1     Running   0          2m
# streamlit-xxxxxxx-xxxxx     1/1     Running   0          2m
# streamlit-xxxxxxx-xxxxx     1/1     Running   0          2m

# Check services
kubectl get svc -n churn-prediction

# View logs
kubectl logs -f deployment/api -n churn-prediction
```

---

## 🔧 Useful Commands

### View Everything
```bash
kubectl get all -n churn-prediction
```

### Scale Up/Down
```bash
# Scale API to 3 replicas
kubectl scale deployment api --replicas=3 -n churn-prediction

# Scale down to 1
kubectl scale deployment api --replicas=1 -n churn-prediction
```

### Update to New Version
```bash
# After pushing new Docker image
kubectl rollout restart deployment/api -n churn-prediction
kubectl rollout restart deployment/streamlit -n churn-prediction
```

### View Logs
```bash
# API logs
kubectl logs -f deployment/api -n churn-prediction

# Streamlit logs
kubectl logs -f deployment/streamlit -n churn-prediction

# All logs
kubectl logs -f -l app=api -n churn-prediction
```

### Describe Resources
```bash
# Get detailed info about deployment
kubectl describe deployment api -n churn-prediction

# Get detailed info about pod
kubectl describe pod <pod-name> -n churn-prediction
```

---

## 🐛 Troubleshooting

### Issue: Pods not starting

```bash
# Check pod status
kubectl get pods -n churn-prediction

# If ImagePullBackOff:
# - Check Docker image exists on Docker Hub
# - Verify image name in k8s/api.yaml and k8s/streamlit.yaml

# If CrashLoopBackOff:
# - Check logs: kubectl logs <pod-name> -n churn-prediction
# - Usually means app is crashing on startup
```

### Issue: Can't access LoadBalancer URL

```bash
# Check if LoadBalancer is ready
kubectl get svc -n churn-prediction

# If <pending>:
# - Wait a few more minutes
# - Check AWS Console → EC2 → Load Balancers

# If still pending after 10 minutes:
# - Your AWS account might not have LoadBalancer permissions
# - Try using NodePort instead (see below)
```

### Issue: Permission denied errors

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Update kubeconfig
aws eks update-kubeconfig --name churn-prediction-cluster --region us-east-1
```

### Alternative: Use NodePort (if LoadBalancer doesn't work)

Edit `k8s/api.yaml` and `k8s/streamlit.yaml`:

Change:
```yaml
spec:
  type: LoadBalancer
```

To:
```yaml
spec:
  type: NodePort
```

Then access via Node IP:
```bash
kubectl get nodes -o wide
# Use EXTERNAL-IP:NodePort
```

---

## 💰 Cost Information

**Approximate AWS costs:**
- EKS Cluster: $0.10/hour (~$73/month)
- 2x t3.medium nodes: $0.0832/hour each (~$60/month each)
- LoadBalancers: $0.0225/hour each (~$16/month each)

**Total: ~$225/month**

**To reduce costs:**
1. Use t3.small instead of t3.medium (~$36/month savings)
2. Use 1 node instead of 2 (~$60/month savings)
3. Delete when not using: `./scripts/cleanup-simple.sh`

---

## 🧹 Cleanup (Delete Everything)

**When you're done:**

```bash
# Make script executable
chmod +x scripts/cleanup-simple.sh

# Run cleanup
./scripts/cleanup-simple.sh

# Then delete EKS cluster in AWS Console:
# https://console.aws.amazon.com/eks
# Select cluster → Delete node group → Delete cluster
```

**Or use CLI:**
```bash
# Delete application
kubectl delete namespace churn-prediction

# Delete cluster
eksctl delete cluster --name churn-prediction-cluster --region us-east-1
```

⚠️ **This will delete everything and stop charges!**

---

## 📋 Complete Checklist

- [ ] AWS CLI installed
- [ ] kubectl installed
- [ ] AWS configured (`aws configure`)
- [ ] EKS cluster created
- [ ] Docker images on Docker Hub
- [ ] Deploy script executed
- [ ] Pods running (check: `kubectl get pods -n churn-prediction`)
- [ ] Services have external IPs
- [ ] Can access API URL
- [ ] Can access Streamlit URL

---

## 🎯 What You Get

After deployment:
- ✅ App running in AWS cloud
- ✅ Accessible from anywhere on internet
- ✅ 2 API pods (high availability)
- ✅ 2 Streamlit pods (high availability)
- ✅ Auto-restart if pods crash
- ✅ LoadBalancer distributes traffic

---

## 🚀 Next Steps

Once deployed:

1. **Test your app** - Visit the URLs
2. **Share the URLs** - Let others test
3. **Update your app**:
   ```bash
   # Build new Docker images
   docker build -t username/churn-prediction-api:latest -f docker/Dockerfile.api .
   docker push username/churn-prediction-api:latest

   # Restart deployment
   kubectl rollout restart deployment/api -n churn-prediction
   ```
4. **Monitor**: Check logs with `kubectl logs`
5. **Scale**: Add more replicas if needed

---

## 💡 Tips

1. **Save money**: Delete resources when not using
2. **Check status**: Use `kubectl get all -n churn-prediction`
3. **View logs**: Use `kubectl logs -f deployment/api -n churn-prediction`
4. **Update app**: Push new Docker image → restart deployment
5. **Get help**: Check AWS EKS documentation

---

## 📚 Common Commands Cheat Sheet

```bash
# Status
kubectl get pods -n churn-prediction
kubectl get svc -n churn-prediction
kubectl get all -n churn-prediction

# Logs
kubectl logs -f deployment/api -n churn-prediction
kubectl logs <pod-name> -n churn-prediction

# Scale
kubectl scale deployment api --replicas=3 -n churn-prediction

# Update
kubectl rollout restart deployment/api -n churn-prediction
kubectl rollout status deployment/api -n churn-prediction

# Describe
kubectl describe pod <pod-name> -n churn-prediction
kubectl describe svc api-service -n churn-prediction

# Delete
kubectl delete namespace churn-prediction
```

---

## 🎉 Success!

When you see your API and Streamlit accessible via URLs, you've successfully deployed to AWS!

Your app is now:
- ✅ Running in the cloud
- ✅ Accessible globally
- ✅ Highly available
- ✅ Auto-healing
- ✅ Production-ready

---

## ❓ Need Help?

**Can't create EKS cluster?**
- Check IAM permissions
- Try AWS Console instead of CLI
- Verify AWS account is active

**Pods not starting?**
- Check image name matches Docker Hub
- View logs: `kubectl logs <pod-name> -n churn-prediction`
- Check if image is public on Docker Hub

**LoadBalancer stuck on pending?**
- Wait 5-10 minutes
- Check AWS Console → EC2 → Load Balancers
- Try NodePort type instead

**Costs too high?**
- Use smaller instance types (t3.small)
- Reduce number of nodes to 1
- Delete when not using

---

**That's it! Simple AWS deployment that actually works.** ☁️

No complex Terraform, just working Kubernetes!
