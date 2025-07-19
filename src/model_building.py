import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm
from sklearn.feature_extraction.text import TfidfVectorizer

# logger configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log/model_building.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file.

    Returns:
        dict: Parameters dictionary.
    """
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters received from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found at %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): URL or path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        df.fillna("", inplace=True)  # Fill any NaN values
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error("File not found at %s", data_url)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame, max_features: int, ngram_range: tuple
) -> tuple:
    """
    Apply TF-IDF vectorization to the 'text' column of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'text' column.
        params (dict): Parameters for TF-IDF vectorization.

    Returns:
        pd.DataFrame: DataFrame with TF-IDF features.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        x_train = train_data["clean_comment"].values
        y_train = train_data["category"].values

        x_train_tfidf = vectorizer.fit_transform(x_train)
        logger.debug(
            "TF-IDF vectorization applied with max_features=%d and ngram_range: %s",
            max_features,
        )
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, "model/tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        logger.debug("Tfidf is applied with trigrams and data transformed")
        return x_train_tfidf, y_train

    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
):
    """Train lgbm model"""
    try:
        best_model = lightgbm.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            force_col_wise=True,
        )
        best_model.fit(X_train, y_train)
        logger.debug("Lgbm model training completed")
        return best_model

    except Exception as e:
        logger.error("Error duning lgbm model training: %s", e)


def save_model(model, file_path: str) -> None:
    """Save the model path
    Args:
        model ([type]): [description]
        file_path (str): [description]
    """
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.debug("Model saved in the %s", file_path)

    except Exception as e:
        logger.error("Error occured while saving the model: %s", e)
        raise


def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

        # load parameters
        params = load_params(os.path.join(root_dir, "params.yaml"))

        max_features = params["model_building"]["max_features"]
        ngram_range = tuple(params["model_building"]["ngram_range"])

        learning_rate = params["model_building"]["learning_rate"]
        max_depth = params["model_building"]["max_depth"]
        n_estimators = params["model_building"]["n_estimators"]

        # load preprocessed data from the csv file

        train_data = pd.read_csv(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../data/interim/train_preprocessed.csv"
                )
            )
        )
        train_data.dropna(inplace=True)
        # Apply TfIDF feature engineering on the loaded data
        X_train_tfidf, y_train = apply_tfidf(
            train_data=train_data, max_features=max_features, ngram_range=ngram_range
        )

        logger.info("TF IDF transformation completed")

        # train the lightGBM model using hyperparameters from yaml file
        best_model = train_lgbm(
            X_train_tfidf, y_train, learning_rate, max_depth, n_estimators
        )

        # save the trained model into the root directory
        save_model(best_model, os.path.join(root_dir, "model/lgbm_model.pkl"))

    except Exception as e:
        logger.error("Exception has occured %s", e)
        print(f"error: ", e)


if __name__ == "__main__":
    main()
