# Phase 5: CI/CD - Simple & Working Guide

## 🎯 What This Does

Automatically runs tests and builds Docker images when you push code to GitHub.

---

## 📦 Files Created (5 files)

1. `.github/workflows/ci.yml` - Runs tests on every push
2. `.github/workflows/docker.yml` - Builds Docker images
3. `.pre-commit-config.yaml` - Local code checks
4. `.flake8` - Linting rules
5. `pyproject.toml` - Python configuration


---

1. .github/workflows/ci.yml

Purpose: Defines a GitHub Actions workflow for Continuous Integration (CI).

Typical Use: Runs automated tests every time code is pushed to the repository.

What it usually contains:

The trigger (e.g., on: push or on: pull_request)

The jobs to run (e.g., testing on different Python versions)

Steps like checking out code, installing dependencies, and running tests (pytest, unittest, etc.)

Why important: Ensures that new changes don’t break the codebase.

2. .github/workflows/docker.yml

Purpose: Defines a GitHub Actions workflow for building Docker images.

Typical Use: Automatically builds Docker images for your application whenever code is pushed or a release is made.

What it usually contains:

Docker build steps (docker build -t <image-name> .)

Optionally, pushing the image to a container registry like Docker Hub or GitHub Container Registry

Why important: Automates containerization so deployments can be consistent and reproducible.

3. .pre-commit-config.yaml

Purpose: Configures pre-commit hooks for local code checks before committing.

Typical Use: Ensures code quality and style automatically before code is even pushed.

What it usually contains:

A list of hooks, like black (code formatting), flake8 (linting), isort (sorting imports), or custom scripts

Hook configuration options, e.g., file types to check

Why important: Prevents style issues and common mistakes from entering the repository.

4. .flake8

Purpose: Configuration file for the Flake8 linter.

Typical Use: Defines Python linting rules like:

Max line length

Ignored error codes

Excluded files or directories

Why important: Ensures consistent code style and catches errors early, especially in teams.

5. pyproject.toml

Purpose: Central configuration file for Python projects.

Typical Use: Can configure:

Build tools (setuptools, poetry)

Code formatters (like black)

Tool-specific settings (e.g., isort, pytest)

Metadata (name, version, dependencies)

Why important: Standardizes project configuration in one place and supports modern Python packaging.


---

## 🚀 Setup (5 minutes)

### Step 1: Create GitHub Repository

```bash
# Initialize git
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Add GitHub Secrets

Go to your GitHub repo:
**Settings → Secrets and variables → Actions → New repository secret**

Add these **2 secrets**:
1. `DOCKER_USERNAME` - Your Docker Hub username
2. `DOCKER_PASSWORD` - Your Docker Hub password

### Step 3: Push Code

```bash
git add .
git commit -m "Setup CI/CD"
git push
```

**That's it!** GitHub Actions will automatically start.

---

## ✅ What Happens Automatically

### On Every Push or Pull Request:

1. **Tests Run** - All your tests execute
2. **Code Quality Check** - Flake8 and Black check code
3. **Docker Build** (PR only) - Test that Docker builds work

### On Push to Main Branch:

1. Everything above, PLUS:
2. **Docker images built** - API, Streamlit, Training
3. **Pushed to Docker Hub** - Tagged as `latest`

### On Version Tag (like v1.0.0):

1. Everything above, PLUS:
2. **Tagged images** - Both `latest` and `v1.0.0`

---

## 📊 View Your CI/CD

After pushing, go to:
```
https://github.com/YOUR_USERNAME/YOUR_REPO/actions
```

You'll see:
- ✅ Green checks = passed
- ❌ Red X = failed
- 🟡 Yellow = running

---

## 🔧 Local Setup (Optional but Recommended)

### Install Pre-commit Hooks

```bash
# Install
pip install pre-commit

# Setup
pre-commit install

# Now every commit automatically checks:
# - Black formatting
# - Flake8 linting
# - Trailing whitespace
# - Large files
```

### Run Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific check
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

---

## 🎨 Add Status Badges to README

Add to your `README.md`:

```markdown
![CI](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/CI%20Pipeline/badge.svg)
![Docker](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Docker%20Build%20and%20Push/badge.svg)
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual values.

---

## 🐛 Troubleshooting

### Issue: Workflow fails with "Secret not found"

**Solution:** Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` in GitHub Secrets.

### Issue: Tests fail in CI but pass locally

**Solution:** Make sure all dependencies are in `requirements.txt`

### Issue: Docker build fails

**Solution:**
1. Check Dockerfile exists: `docker/Dockerfile.api`
2. Test locally: `docker build -f docker/Dockerfile.api .`
3. Check GitHub Secrets are correct

### Issue: Pre-commit hooks fail

**Solution:**
```bash
# Auto-fix most issues
black src/ api/ streamlit_app/

# Then commit again
git add .
git commit -m "Fix formatting"
```

---

## 📝 Workflow Files Explained

### ci.yml (Main CI)

```yaml
on:
  push:              # Run on push to main/develop
  pull_request:      # Run on pull requests

jobs:
  test:             # Run tests
  lint:             # Check code quality
  docker-build:     # Test Docker build (PR only)
```

### docker.yml (Docker)

```yaml
on:
  push:
    branches: [main]  # Only on main branch
    tags: ['v*']      # Or version tags

jobs:
  build-api:         # Build API image
  build-streamlit:   # Build Streamlit image
  build-training:    # Build Training image
```

---

## 🔄 Typical Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-model

# 2. Make changes
# Edit files...

# 3. Commit (pre-commit runs automatically)
git commit -m "Add new model"

# 4. Push
git push origin feature/new-model

# 5. Create PR on GitHub
# CI runs automatically:
# - Tests
# - Linting
# - Docker build test

# 6. Merge PR
# After merge to main:
# - Everything runs again
# - Docker images built and pushed

# 7. Create release tag
git tag v1.0.0
git push origin v1.0.0

# - Images tagged as v1.0.0 and latest
```

---

## ✅ Success Checklist

- [ ] GitHub repository created
- [ ] Secrets added (`DOCKER_USERNAME`, `DOCKER_PASSWORD`)
- [ ] Code pushed to GitHub
- [ ] CI workflow running (check Actions tab)
- [ ] Tests passing (green check)
- [ ] Docker images on Docker Hub
- [ ] Pre-commit hooks installed locally (optional)
- [ ] Status badges added to README (optional)

---

## 🎯 What's Next?

Once CI/CD is working:

1. **Every push triggers tests** - No broken code gets merged
2. **Docker images auto-build** - Always up to date
3. **Easy deployments** - Just push to deploy
4. **Version tracking** - Tag releases with versions

---

## 💡 Tips

1. **Small commits** - Easier to debug if CI fails
2. **Run tests locally** - Before pushing
3. **Check Actions tab** - If something fails
4. **Fix formatting** - Run `black` before committing
5. **Use branches** - Don't commit directly to main

---

## 📚 Common Commands

```bash
# Check if code is formatted
black --check src/ api/

# Format code
black src/ api/

# Run linting
flake8 src/ api/

# Run tests
pytest

# Install pre-commit
pip install pre-commit
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Push to GitHub
git push

# Create version tag
git tag v1.0.0
git push origin v1.0.0
```

---

## 🎉 Success!

When you see green checks in GitHub Actions, everything is working!

Your code is now:
- ✅ Automatically tested
- ✅ Quality checked
- ✅ Built into Docker images
- ✅ Ready for deployment

---

## ❓ Need Help?

**Common Questions:**

**Q: Where do I add secrets?**
A: GitHub repo → Settings → Secrets and variables → Actions

**Q: Why isn't CI running?**
A: Check that files are in `.github/workflows/` directory

**Q: Tests fail in CI but work locally?**
A: Make sure `requirements.txt` has all dependencies

**Q: How do I see what failed?**
A: Click on the red X in GitHub → Click on failed job → See logs

---

**That's it! Simple CI/CD that actually works.** 🚀

No complex configurations, just working automation.
