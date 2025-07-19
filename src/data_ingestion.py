import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

# logger configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log/data_ingestion.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    load parameters from yaml file
    Args:
        params_path (str): file path
    Returns:
        dict: parameters dictionary
    """

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters receved from %s", params_path)
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
    Load data from CSV files
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded drom %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
    """data preprocessing
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Cleaned and structured dataframe
    """
    try:
        df.dropna(inplace=True)  ## remove nan values
        df.drop_duplicates(inplace=True)  ## remove duplicates
        df = df[df["clean_comment"].str.strip() != ""]  # remove rows with empty string
        logger.debug(
            "Data Preprocessing completed: Missing Valued, Duplicate and empty strings removed"
        )
        return df
    except KeyError as e:
        logger.error("Missing key Column in the dataframe: %s", e)
        raise

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str
) -> None:
    """Saving the data at a path

    Args:
        train_data (pd.DataFrame): [description]
        test_data (pd.DataFrame): [description]
        data_path (str): [description]
    """

    try:
        # Create the data/raw directory if it does not exist
        print(data_path)
        os.makedirs(data_path, exist_ok=True)

        # save train and test data ino seprate csv files
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

        logger.debug("Data saved at %s", data_path)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def main():
    try:
        # load parameters from the params.yaml in the root directory
        params = load_params(
            params_path=os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "params.yaml",
            )
        )

        test_size = params["data_ingestion"]["test_size"]
        # Print the directory path one levels back from this file
        one_levels_back = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        df = load_data(os.path.join(one_levels_back, "data", "reddit.csv"))
        df = preprocessing_data(df)

        # spilit the training an dtest data
        train_data, test_data = train_test_split(
            df, test_size=test_size, random_state=42
        )

        # save the data
        save_data(
            train_data,
            test_data,
            data_path=os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "data",
                "raw",
            ),
        )

        print("Data ingestion completed successfully")

    except Exception as e:
        logger.error("An error occurred in the main function: %s", e)


if __name__ == "__main__":
    main()
    logger.info("Data ingestion script started")
