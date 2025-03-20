import logging
import joblib
from typing import Dict
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.model import inference

# Initialize FastAPI app
app = FastAPI()

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Set Model Path
MODEL_PATH = './model'


def load_model(model_path):
    """
    Load the trained model, encoder, and label binarizer from the specified directory.

    Inputs
    ------
    model_path : str
        Path to the directory containing the saved model, encoder, and label binarizer.

    Returns
    -------
    model : Trained model
    encoder : Trained OneHotEncoder
    lb : Trained LabelBinarizer
    """
    try:
        model = joblib.load(f"{model_path}/trained_model.pkl")
        encoder = joblib.load(f"{model_path}/encoder.pkl")
        lb = joblib.load(f"{model_path}/label_binarizer.pkl")

        logger.info('Model, encoder, and label binarizer loaded successfully.')

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model, encoder, lb = None, None, None  # Prevents crashes on startup

    return model, encoder, lb


model, encoder, lb = load_model(MODEL_PATH)


# Example endpoint to check if the app is working
@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my FastAPI application!"}


# Define input schema using Pydantic (Handles hyphenated column names)
class InferenceRequest(BaseModel):
    age: int = Field(..., alias="age")
    workclass: str = Field(..., alias="workclass")
    fnlgt: int = Field(..., alias="fnlgt")
    education: str = Field(..., alias="education")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        populate_by_name = True  # Allows using alias names in request bodies
        schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 123456,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married",
                "occupation": "Tech",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# POST endpoint for model inference
@app.post("/predict", response_model_by_alias=True)
def predict(request: InferenceRequest) -> Dict[str, str]:
    """
    Run inference using the trained model.

    Inputs
    ------
    request : InferenceRequest
        JSON payload containing the input data.

    Returns
    -------
    response : dict
        JSON response containing predictions.
    """
    if model is None:
        return {"error": "Model not loaded. Check logs for issues."}

    # Convert input data into dictionary
    data_dict = request.dict()

    # Extract categorical and numerical features separately
    categorical_values = [data_dict[feature] for feature in [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]]  # Only categorical fields

    numerical_values = [data_dict[feature] for feature in [
        "age",
        "fnlgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week"
    ]]  # Only numerical fields

    # Convert lists to NumPy arrays
    categorical_array = np.array([categorical_values])
    numerical_array = np.array([numerical_values])

    # Encode categorical features
    categorical_encoded = encoder.transform(categorical_array)

    # Concatenate numerical and encoded categorical features
    X_input_encoded = np.concatenate([numerical_array, categorical_encoded], axis=1)

    # Run inference
    preds = inference(model, X_input_encoded)

    # Decode predictions (if using LabelBinarizer)
    decoded_preds = lb.inverse_transform(preds)

    return {"prediction": decoded_preds[0]}
