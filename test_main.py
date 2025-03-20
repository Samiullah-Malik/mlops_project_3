from fastapi.testclient import TestClient

from main import app  # Import FastAPI app from main.py

# Create a test client
client = TestClient(app)


# Test Case 1: GET /
def test_get_root():
    """Test the GET method on the root endpoint."""
    response = client.get("/")

    # Check if the response is successful
    assert response.status_code == 200

    # Check if response contains expected message
    expected_message = {"message": "Welcome to my FastAPI application!"}
    assert response.json() == expected_message


# Test Case 2: POST /predict - Low-income Prediction (<=50K)
def test_predict_low_income():
    """Test inference for a profile expected to predict '<=50K'."""

    payload = {
        "age": 22,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=payload)

    # Check if the response is successful
    assert response.status_code == 200

    # Check if prediction is '<=50K'
    assert response.json()["prediction"] == "<=50K"


# Test Case 3: POST /predict - High-income Prediction (>50K)
def test_predict_high_income():
    """Test inference for a profile expected to predict '>50K'."""

    payload = {
        "age": 45,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 300000,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 20000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=payload)

    # Check if the response is successful
    assert response.status_code == 200

    # Check if prediction is '>50K'
    assert response.json()["prediction"] == ">50K"
