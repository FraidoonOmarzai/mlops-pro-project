import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

dirs = [
    "src",
    os.path.join("src", "components"),
    os.path.join("src", "utils"),
    os.path.join("src", "pipeline"),  
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebook",
    "configs",
    "mlruns",
    "artifacts",
    os.path.join("artifacts", "models"),
    os.path.join("artifacts", "prepocessors"),
    os.path.join("artifacts", "metrics"),
    "tests",
    os.path.join("tests", "unit"),
    os.path.join("tests", "integration"),
    os.path.join("tests", "data"),
    "logs",
    "scripts",
    
]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    # To get Git to recognize an empty directory, the unwritten rule is to put a file named .gitkeep in it
    with open(os.path.join(dir_, ".gitkeep"), "w") as f:
        logging.info(f"Creating directory:{dir_}")
        pass


files = [
    "Dockerfile",
    "setup.py",
    "app.py",
    "requirements.txt",
    '.env',
    os.path.join("notebook", "experiments.ipynb"),
    os.path.join("src", "__init__.py"),
    os.path.join("src", "pipeline", "__init__.py"),
    os.path.join("src", "components", "__init__.py"),
    os.path.join("src", "utils", "__init__.py"),
    os.path.join("configs", "config.yaml"),
    os.path.join("configs", "model_config.yaml"),
    os.path.join("tests", "__init__.py"),
    os.path.join("src", "logger.py"),
    os.path.join("src", "exception.py"),
]

for file_ in files:
    with open(file_, "w") as f:
        logging.info(f"Creating file: {file_}")
        pass