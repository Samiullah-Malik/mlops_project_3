# Script to train machine learning model.
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
def load_data(data_path):
    return pd.read_csv(data_path)

data_path='../data/census.csv'
data = load_data(data_path)
logger.info(f'Loaded data: {data_path}')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
logger.info('Split data into train and test')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)
logger.info('Process the test data')

# Train and save a model.
model = train_model(X_train, y_train)
logger.info('Train the model')

def save_model(model, encoder, lb, model_path):
    joblib.dump(model, f"{model_path}/trained_model.pkl")
    joblib.dump(encoder, f"{model_path}/encoder.pkl")
    joblib.dump(lb, f"{model_path}/label_binarizer.pkl")

save_model(model, encoder, lb, '../model')
logger.info('Save the model')