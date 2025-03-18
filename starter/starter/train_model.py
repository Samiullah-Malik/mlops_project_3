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

# Run Model inference
y_preds = inference(model, X_test)

# Compute model performance metrics
def compute_model_metrics_per_slice(df, feature, y_true, y_preds, output_file):
    """
    Computes model performance metrics (precision, recall, F1-score) for each unique value of a given categorical feature.

    Inputs
    ------
    df : pd.DataFrame
        The dataset containing the categorical features.
    feature : str
        The categorical feature to compute performance slices for.
    y_true : np.array
        Known labels, binarized.
    y_pred : np.array
        Predicted labels, binarized.

    Returns
    -------
    None (Prints performance metrics per slice)
    """
    logger.info(type(df))
    unique_values = df[feature].unique()  # Get all unique values of the feature
    

    with open(output_file, 'a') as f:
        f.write(f"\nPerformance metrics for feature: {feature}\n" + "-" * 50)
        f.write("\n")

        for value in unique_values:
            # Filter rows where the feature has a specific value
            mask = df[feature] == value
            y_true_slice = y_true[mask]
            y_pred_slice = y_preds[mask]

            # Ensure we have data points for this slice
            if len(y_true_slice) == 0:
                continue

            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

            # Print results
            print(f"Feature Value: {value}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-score:  {fbeta:.4f}")
            print("-" * 50)


            f.write(f"Feature Value: {value}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1-score:  {fbeta:.4f}\n")
            f.write("-" * 50)
            f.write("\n")

        f.write("\n")



logger.info("Computing Model Performance On Slices of Data")
# with open('slice_output.txt', 'w'):
for feature in cat_features:
    compute_model_metrics_per_slice(test, feature, y_test, y_preds, 'slice_output.txt')
    