from flask import Flask, jsonify, request
import mlflow
from flask_cors import CORS
from data_preprocessing import preprocessing_data
import pickle

app = Flask(__name__)
CORS(app=app)



def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://localhost:5000")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def load_model(vectorizer_path, model_path):
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        return model, vectorizer

    except Exception as e:
        return "No files found"


model, vectorizer = load_model(
    "/home/anzo/ComputerVision/ML/Project/SentimentAnalysis/model/tfidf_vectorizer.pkl",
    "/home/anzo/ComputerVision/ML/Project/SentimentAnalysis/model/lgbm_model.pkl",
)

if isinstance(model, str) or isinstance(vectorizer, str):
    model = None
    vectorizer = None


@app.route("/")
def home():
    return " Welcome to the ML Testing API"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comments = data.get("comments") if data else None

    if not comments:
        return jsonify({"error": "No comment provided"}), 400
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded"}), 500

        preprocessed_comments = [preprocessing_data(comment) for comment in comments]

        transformed_comments = vectorizer.transform(preprocessed_comments)

        dense_comments = transformed_comments.toarray()

        # FIX: Convert to DataFrame with correct feature names
        import pandas as pd

        feature_names = vectorizer.get_feature_names_out()
        input_df = pd.DataFrame(dense_comments, columns=feature_names)

        predictions = model.predict(input_df).tolist()

        print(predictions)
        response = [{"comment": comment, "sentiment": sentimet} for comment, sentimet in zip(comments, predictions)]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
      

def main():
    app.run(host="0.0.0.0", port=5001, debug=True)

if __name__=="__main__":
    main()