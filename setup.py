import setuptools


__version__ = "0.0.1"

SRC_REPO = "MLOps_Project"
AUTHOR_USER_NAME = "FraidoonOmarzai"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="Implementation of end-to-end MLOps Project",
    packages=setuptools.find_packages()
)