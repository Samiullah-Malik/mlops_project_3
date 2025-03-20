import requests
import json
import pyautogui

# Define the endpoint URL
dev_url = "http://127.0.0.1:8000/predict"
prod_url = "https://mlops-project-3-f4b491fde494.herokuapp.com/predict"

# Define the data to be sent in the POST request
data = {
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

# Make the POST request to the API
response = requests.post(prod_url, json=data)

# Print the response content and status code
print("Status Code:", response.status_code)
print("Response Body:", response.json())