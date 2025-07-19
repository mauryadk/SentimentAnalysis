import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging


# logger configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log/data_preprocessing.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# download required nltk data
nltk.download("stopwords")
nltk.download("wordnet")

os.makedirs("data/interim", exist_ok=True)

# define the preprocessing function


def preprocessing_data(comment):
    """data preprocessing
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Cleaned and structured dataframe
    """
    try:
        comment = comment.lower()  # make it lower case

        comment.strip()  # remove white spaces

        # remove newline charecters
        comment = re.sub(r"\n", " ", comment)

        # remove non-alphanumeric chracters except punctuation
        comment = re.sub(r"[^a-zA-Z0-9\s]+", "", comment)

        # Remove stop words except important for sentiment analysis
        stop_words = set(stopwords.words("english")) - {
            "not",
            "no",
            "nor",
            "don",
            "don't",
            "but",
            "against",
            "aren",
            "aren't",
            "couldn",
            "couldn't",
            "didn",
            "didn't",
        }
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # lemmatize te words
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def normalize_text(df):
    """Apply preprocessing to the text data present tin the dataframe
    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
        pd.DataFrame: Cleaned and structured dataframe
    ):
    """

    try:
        df["clean_comment"] = df["clean_comment"].apply(preprocessing_data)
        logger.debug("Text preprocessing completed")
        return df

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
        logger.debug(
            "Creating the data/raw directory if it does not exist %s", data_path
        )
        # Create the data/raw directory if it does not exist

        os.makedirs(data_path, exist_ok=True)
        logger.debug(f"Directory created at {data_path} or already exists", data_path)
        # save train and test data ino seprate csv files
        train_data.to_csv(
            os.path.join(data_path, "train_preprocessed.csv"), index=False
        )
        test_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)
        logger.debug("Data saved at %s", data_path)

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


def main():
    try:
        logger.debug("Data preprocessing script started")
        # load the data from raw
        train_data_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "raw",
            "train.csv",
        )
        test_data_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "raw",
            "test.csv",
        )


        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        logger.debug("Data loaded from raw directory")

        # normalize the text data
        train_data = normalize_text(train_data)

        test_data = normalize_text(test_data)
        print("Test1")
        logger.debug("Text preprocessing completed")

        save_data(
            train_data=train_data,
            test_data=test_data,
            data_path=os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "data",
                "interim",
            ),
        )
        logger.debug("Data preprocessing completed successfully")

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
    logger.info("Data preprocessing script started")
