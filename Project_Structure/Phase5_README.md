# Phase 5: CI/CD with GitHub Actions - Complete Guide

## ⚙️ Overview

Phase 5 implements comprehensive CI/CD pipelines using GitHub Actions for automated testing, building, security scanning, and deployment.

---

## 📦 What's Been Created

### **GitHub Actions Workflows (6)**
1. **CI Pipeline** (`ci.yml`) - Automated testing & validation
2. **Docker Build** (`docker-build.yml`) - Build & push Docker images
3. **Code Quality** (`code-quality.yml`) - Linting & formatting
4. **Security** (`security.yml`) - Vulnerability scanning
5. **Deployment** (`deploy.yml`) - AWS EKS deployment
6. **Release** (`release.yml`) - Automated releases

### **Configuration Files (4)**
7. `.pre-commit-config.yaml` - Pre-commit hooks
8. `.flake8` - Flake8 linting config
9. `pyproject.toml` - Black, isort, mypy config
10. GitHub issue/PR templates

---

## 🚀 Quick Start

### **Step 1: Setup GitHub Repository**
```bash
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

### **Step 2: Configure GitHub Secrets**

Go to: **Settings → Secrets and variables → Actions**

Add these secrets:
- `DOCKER_USERNAME` - Your Docker Hub username
- `DOCKER_PASSWORD` - Your Docker Hub password/token
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `SLACK_WEBHOOK_URL` - (Optional) For notifications

### **Step 3: Enable GitHub Actions**

GitHub Actions should be enabled by default. Verify at:
**Settings → Actions → General**

### **Step 4: Install Pre-commit Hooks (Local)**
```bash
pip install pre-commit
pre-commit install
```

### **Step 5: Make Your First Commit**
```bash
git add .
git commit -m "Setup CI/CD pipeline"
git push
```

GitHub Actions will automatically trigger! 🎉

---

## 📊 CI/CD Pipeline Overview

```
┌─────────────────────────────────────────────┐
│           Push/PR to main/develop           │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
  ┌──────────┐         ┌──────────┐
  │   Test   │         │   Lint   │
  └────┬─────┘         └────┬─────┘
       │                    │
       └──────────┬─────────┘
                  ↓
           ┌─────────────┐
           │  Security   │
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │ Docker Build│
           └──────┬──────┘
                  │
         ┌────────▼────────┐
         │   Deploy (Tag)  │
         └─────────────────┘
```

---

## 🔧 Workflow Details

### **1. CI Pipeline** (`ci.yml`)

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Jobs:**
- ✅ Run tests (Python 3.9, 3.10, 3.11)
- ✅ Check code quality (flake8, black, isort)
- ✅ Security scanning (Bandit, Safety)
- ✅ Test Docker build
- ✅ Upload coverage to Codecov

**Usage:**
```bash
# Automatically runs on push
git push origin main

# View results
https://github.com/youruser/yourrepo/actions
```

---

### **2. Docker Build & Push** (`docker-build.yml`)

**Triggers:**
- Push to `main`
- Tags (`v*`)
- Manual trigger
- Pull requests (build only, no push)

**Jobs:**
- ✅ Build API, Streamlit, Training images
- ✅ Multi-arch builds (amd64, arm64)
- ✅ Push to Docker Hub with tags
- ✅ Scan images with Trivy
- ✅ Verify images
- ✅ Cleanup old images

**Tags Created:**
- `latest` - Latest main branch
- `v1.0.0` - Specific version
- `main-abc123` - Git commit SHA
- `pr-42` - Pull request number

---

### **3. Code Quality** (`code-quality.yml`)

**Checks:**
- ✅ Black formatting
- ✅ isort import sorting
- ✅ Flake8 linting
- ✅ Pylint analysis
- ✅ MyPy type checking
- ✅ Cyclomatic complexity
- ✅ Docstring coverage
- ✅ Dependency licenses

**Auto-fix:**
- Automatically fixes formatting on PRs
- Commits fixes back to branch

---

### **4. Security Scanning** (`security.yml`)

**Scans:**
- ✅ Dependency vulnerabilities (Safety, pip-audit)
- ✅ Code security (Bandit, Semgrep)
- ✅ Secret detection (Gitleaks, TruffleHog)
- ✅ SAST (CodeQL)
- ✅ Docker image scanning (Trivy, Grype)
- ✅ Compliance checks

**Schedule:**
- Runs on every push
- Weekly full scan (Sunday midnight)

---

### **5. Deployment** (`deploy.yml`)

**Environments:**
- **Staging** - Automatic on tags
- **Production** - Manual approval required

**Strategy:**
- Blue-Green deployment
- Health checks
- Automatic rollback on failure

**Process:**
1. Configure AWS credentials
2. Update kubeconfig
3. Deploy green environment
4. Run health checks
5. Switch traffic
6. Monitor
7. Scale down blue

**Trigger:**
```bash
# Tag for production deploy
git tag v1.0.0
git push origin v1.0.0

# Manual deploy to staging
# Go to Actions → Deploy → Run workflow
```

---

### **6. Release Automation** (`release.yml`)

**Triggers:**
- New tag (`v*`)
- Manual trigger

**Actions:**
- ✅ Generate changelog
- ✅ Create GitHub release
- ✅ Publish documentation
- ✅ Send notifications

**Creating a Release:**
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

---

## 🛠️ Local Development

### **Pre-commit Hooks**

Install hooks:
```bash
pip install pre-commit
pre-commit install
```

Run manually:
```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

### **Code Formatting**

```bash
# Format with Black
black src/ api/ streamlit_app/

# Sort imports
isort src/ api/ streamlit_app/

# Lint with Flake8
flake8 src/ api/

# Type check
mypy src/ api/
```

### **Run Tests Locally**

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov=api --cov-report=html

# Specific category
pytest -m unit
pytest -m integration
```

---

## 📈 Monitoring CI/CD

### **View Workflow Runs**
```
https://github.com/youruser/yourrepo/actions
```

### **Status Badges**

Add to README.md:
```markdown
![CI](https://github.com/youruser/yourrepo/workflows/CI%20Pipeline/badge.svg)
![Docker](https://github.com/youruser/yourrepo/workflows/Docker%20Build%20%26%20Push/badge.svg)
![Security](https://github.com/youruser/yourrepo/workflows/Security%20Scanning/badge.svg)
[![codecov](https://codecov.io/gh/youruser/yourrepo/branch/main/graph/badge.svg)](https://codecov.io/gh/youruser/yourrepo)
```

### **Codecov Integration**

1. Sign up at https://codecov.io
2. Add repository
3. Badge automatically generated

---

## 🔐 Security Best Practices

### **Secrets Management**
- ✅ Never commit secrets to code
- ✅ Use GitHub Secrets for sensitive data
- ✅ Rotate secrets regularly
- ✅ Use different secrets for staging/production

### **Dependency Management**
- ✅ Pin dependency versions
- ✅ Run security scans weekly
- ✅ Update dependencies regularly
- ✅ Review dependabot PRs

### **Code Scanning**
- ✅ Enable CodeQL
- ✅ Review security alerts
- ✅ Fix high-severity issues immediately

---

## 🚨 Troubleshooting

### **Issue: Workflow fails on first run**

**Solution:**
```bash
# Ensure all directories exist
mkdir -p artifacts/models artifacts/preprocessors
mkdir -p logs data/raw mlruns

# Commit and push
git add .
git commit -m "Add required directories"
git push
```

### **Issue: Docker build fails**

**Solution:**
Check GitHub Secrets are set:
- DOCKER_USERNAME
- DOCKER_PASSWORD

### **Issue: Tests fail in CI but pass locally**

**Solution:**
```bash
# Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pytest
```

### **Issue: Pre-commit hooks fail**

**Solution:**
```bash
# Update hooks
pre-commit autoupdate

# Run and auto-fix
pre-commit run --all-files
```

---

## 📊 Workflow Status Checks

### **Required Checks** (Branch Protection)

Configure at: **Settings → Branches → Branch protection rules**

Required status checks:
- ✅ Test (Python 3.10)
- ✅ Lint
- ✅ Security
- ✅ Build

### **Pull Request Requirements**
- ✅ All checks must pass
- ✅ At least 1 approval (optional)
- ✅ Up to date with base branch

---

## 🎯 Deployment Strategy

### **Staging Deployment**
- Automatic on any tag
- No approval required
- Used for testing

### **Production Deployment**
- Requires manual approval
- Blue-green strategy
- Automatic health checks
- Rollback on failure

### **Rollback Process**
```bash
# Automatic on failure
# Or manual:
kubectl rollout undo deployment/api -n production
```

---

## 📚 Additional Resources

### **GitHub Actions Docs**
- https://docs.github.com/en/actions

### **Best Practices**
- https://docs.github.com/en/actions/security-guides

### **Marketplace**
- https://github.com/marketplace?type=actions

---

## ✅ Phase 5 Checklist

- ✅ GitHub repository created
- ✅ GitHub Secrets configured
- ✅ CI pipeline running
- ✅ Docker builds working
- ✅ Security scans enabled
- ✅ Pre-commit hooks installed
- ✅ Code quality checks passing
- ✅ Tests passing in CI
- ✅ Docker images pushed to Hub
- ✅ Deployment workflow configured
- ✅ Status badges added to README

---

## 🎉 Success Criteria

Phase 5 is complete when:
1. ✅ All workflows trigger correctly
2. ✅ Tests pass in CI
3. ✅ Docker images build and push
4. ✅ Security scans run
5. ✅ Code quality checks pass
6. ✅ Pre-commit hooks working
7. ✅ Deployments successful
8. ✅ Status badges showing

---

**Phase 5 Complete! Your MLOps project has full CI/CD automation!** 🚀

Next: Deploy to AWS EKS (Phase 6) or maintain and iterate on your pipeline.
