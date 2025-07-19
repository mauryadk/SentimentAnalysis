import os
import sys
from setuptools import find_packages, setup

# Automatically set PYTHONPATH to include the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variable for the project
os.environ["PYTHONPATH"] = project_root

setup(
    name="SentimentAnalysisProject",
    version="0.0.1",
    author="Deepak Maurya",
    author_email="deepakmaurya3296@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["flask", "flask-cors", "mlflow", "scikit-learn", "pandas"],
    entry_points={"console_scripts": ["sentiment-api=api:main"]},
)
