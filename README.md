# Sentiment Analysis Project

A comprehensive machine learning project for sentiment analysis using Reddit comments data. This project implements a complete ML pipeline from data ingestion to model deployment with API endpoints.

## 🎯 Project Overview

This project performs sentiment analysis on Reddit comments using a LightGBM classifier with TF-IDF vectorization. It includes a complete ML pipeline with data preprocessing, model training, evaluation, and deployment via a Flask API.

## ✨ Features

- **Complete ML Pipeline**: End-to-end machine learning workflow
- **Data Version Control**: DVC integration for data and model versioning
- **Experiment Tracking**: MLflow integration for experiment management
- **REST API**: Flask-based API for real-time sentiment prediction
- **Automated Pipeline**: DVC pipeline for reproducible workflows
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Model Registration**: Automatic model versioning and registration

## 🏗️ Project Structure

```
SentimentAnalysis/
├── data/                   # Data files
│   ├── reddit.csv         # Raw dataset
│   └── raw/               # Processed train/test splits
├── src/                   # Source code
│   ├── data_ingestion.py  # Data loading and splitting
│   ├── data_preprocessing.py # Text preprocessing
│   ├── model_building.py  # Model training
│   ├── model_evaluation.py # Model evaluation
│   ├── register_model.py  # Model registration
│   └── api.py            # Flask API
├── model/                 # Trained models
├── runs/                  # MLflow experiment runs
├── log/                   # Log files
├── dvc.yaml              # DVC pipeline configuration
├── params.yaml           # Model parameters
├── requirements.txt      # Python dependencies
└── setup.py             # Package setup
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Git
- DVC (for data version control)
- MLflow (for experiment tracking)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd SentimentAnalysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**

   ```bash
   pip install -e .
   ```

4. **Initialize DVC (if not already done)**
   ```bash
   dvc init
   ```

## 📊 Data

The project uses Reddit comments data for sentiment analysis. The dataset should be placed in the `data/` directory as `reddit.csv`.

### Data Format

- `clean_comment`: The text comment to analyze
- `sentiment`: Target variable (sentiment labels)

## 🔧 Usage

### Running the Complete Pipeline

1. **Execute the DVC pipeline**

   ```bash
   dvc repro
   ```

   This will run the complete pipeline:

   - Data ingestion and splitting
   - Data preprocessing
   - Model building
   - Model evaluation
   - Model registration

### Individual Pipeline Stages

You can run individual stages:

```bash
# Data ingestion
dvc run data_ingestion

# Data preprocessing
dvc run data_preprocessing

# Model building
dvc run model_building

# Model evaluation
dvc run model_evaluation

# Model registration
dvc run model_registration
```

### Running the API

1. **Start the Flask API server**

   ```bash
   python src/api.py
   ```

   The API will be available at `http://localhost:5001`

## 🌐 API Documentation

### Endpoints

#### GET `/`

- **Description**: Welcome message
- **Response**: `"Welcome to the ML Testing API"`

#### POST `/predict`

- **Description**: Predict sentiment for given comments
- **Request Body**:
  ```json
  {
    "comments": ["This is a great product!", "I hate this service"]
  }
  ```
- **Response**:
  ```json
  [
    {
      "comment": "This is a great product!",
      "sentiment": "positive"
    },
    {
      "comment": "I hate this service",
      "sentiment": "negative"
    }
  ]
  ```

### Example Usage

```python
import requests

# API endpoint
url = "http://localhost:5001/predict"

# Sample comments
comments = ["This is amazing!", "I don't like this", "It's okay"]

# Make prediction request
response = requests.post(url, json={"comments": comments})
predictions = response.json()

for pred in predictions:
    print(f"Comment: {pred['comment']}")
    print(f"Sentiment: {pred['sentiment']}\n")
```

## ⚙️ Configuration

### Model Parameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.2
  random_state: 42

model_building:
  model_name: "sentiment_model"
  ngram_range: [1, 3]
  max_features: 1000
  learning_rate: 0.01
  max_depth: 20
  n_estimators: 367
```

### DVC Pipeline (`dvc.yaml`)

The pipeline is configured to run the following stages:

1. **Data Ingestion**: Load and split data
2. **Data Preprocessing**: Clean and preprocess text
3. **Model Building**: Train LightGBM model with TF-IDF
4. **Model Evaluation**: Evaluate model performance
5. **Model Registration**: Register model with MLflow

## 📈 Model Performance

The model uses:

- **Algorithm**: LightGBM Classifier
- **Vectorization**: TF-IDF with n-gram range [1,3]
- **Features**: 1000 maximum features
- **Evaluation**: Confusion matrix and classification metrics

## 🔍 Monitoring and Logging

- **Logs**: Stored in `log/` directory
- **Experiments**: Tracked with MLflow
- **Model Artifacts**: Stored in `runs/` directory
- **Performance Metrics**: Confusion matrix and evaluation reports

## 🛠️ Development

### Adding New Features

1. Create new Python modules in `src/`
2. Update `dvc.yaml` for new pipeline stages
3. Add parameters to `params.yaml` if needed
4. Update tests and documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Implement proper error handling

## 📝 License

[Add your license information here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue in the repository.

---

**Note**: Make sure to have the required data file (`reddit.csv`) in the `data/` directory before running the pipeline.
