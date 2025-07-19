import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import logging

logger = logging.getLogger("model_registration")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("ERROR")

file_handler = logging.FileHandler("log/model_registration_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """load model info from the json file path
    Args:
        file_path (str): _description_
    Returns:
        dict: _description_
    """
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError as e:
        logger.error("File not found")
    except Exception as e:
        logger.error("Unknown error has occured")
        print(e)


def register_model(model_name: str, model_info: dict):
    """Register the model to the mlflow registry"""
    try:
        # Use the correct run_id and artifact_path
        model_uri = f"runs:/{model_info['run_id']}/lgbm_model"
        print("MODEL URI ", model_uri)
        model_version = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )
        logger.debug(
            f"Model {model_name}, and {model_version} registered and transitioned to Staging"
        )
    except Exception as e:
        logger.error(" Error during model registration %s", e)
        raise


def main():
    try:
        mlflow.set_tracking_uri("http://localhost:5000/")

        model_info_path = "model/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "Youtube chrome plugin model"

        register_model(model_name, model_info)

    except Exception as e:
        logger.error("Failed to complete the model registration process: %s", e)
        print(f"error {e}")


if __name__ == "__main__":
    main()
