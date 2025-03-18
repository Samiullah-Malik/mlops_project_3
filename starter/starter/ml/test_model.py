import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .model import train_model, compute_model_metrics, inference

# Set up test data
@pytest.fixture
def synthetic_data():
    """ Generate synthetic training and test data for testing """
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, 100)  # Binary classification (0 or 1)
    X_test = np.random.rand(20, 10)  # 20 test samples
    y_test = np.random.randint(0, 2, 20)  # True labels for test data

    model = train_model(X_train, y_train)
    y_preds = inference(model, X_test)

    return model, X_test, y_test, y_preds

# Test if train_model returns a trained model instance
def test_train_model(synthetic_data):
    model, _, _, _ = synthetic_data
    assert isinstance(model, RandomForestClassifier)

# Test if compute_model_metrics returns three float values
def test_compute_model_metrics(synthetic_data):
    _, _, y_test, y_preds = synthetic_data
    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

# Test if inference output is a NumPy array of expected shape
def test_inference_output(synthetic_data):
    _, X_test, _, y_preds = synthetic_data
    assert isinstance(y_preds, np.ndarray)
    assert y_preds.shape == (X_test.shape[0],)  # Should match the number of test samples
