# Model Card

For additional information see the Model Card paper: [Model Cards for Model Reporting](https://arxiv.org/pdf/1810.03993.pdf)

---

## Model Details

The model was created as part of a machine learning pipeline to predict whether an individual's income is above or below $50K per year. The model uses a **Random Forest Classifier** with default hyperparameters from **scikit-learn 1.3.2**. The model was trained using a Python 3.9 environment with **FastAPI** as the API framework to expose inference as a REST endpoint.

### Project Context:
In this project, the objective was to apply skills acquired in the course to develop a classification model on publicly available **Census Bureau data**. The project aimed to showcase the end-to-end process of model development, deployment, and inference through an API.

### Creator:
- Name: Samiullah Malik

### Libraries Used:
- python 3.94
- scikit-learn 1.3.2
- pandas
- numpy
- FastAPI 0.63.0
- uvicorn
- gunicorn
- joblib

---

## Intended Use

The model is intended to predict an individual's income category (either `<=50K` or `>50K`) based on various socio-economic factors. It is designed to be used by data scientists, engineers, and backend developers as a demonstration of machine learning model deployment and inference through FastAPI.

### Use Cases:
- Predicting income category for demographic analysis.
- Demonstrating the integration of ML models with FastAPI.
- Serving as an example of a machine learning pipeline.

### Limitations:
- The model is not guaranteed to be accurate for populations outside the original dataset's distribution.
- Predictions should not be used for critical decision-making without further validation.

---

## Training Data

The model was trained on the **Census Income Data** obtained from the **UCI Machine Learning Repository**. The original dataset contains **32,563 rows** and **15 columns**, including both numerical and categorical attributes. The target label is the `salary`, which is either `<=50K` or `>50K`.

### Preprocessing:
- Categorical features were one-hot encoded using `OneHotEncoder`.
- Numerical features were left as-is.
- The target label (`salary`) was binarized using `LabelBinarizer`.

### Data Split:
- The dataset was split into **80% training** and **20% testing** using `train_test_split`.

### Categorical Features:
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

### Numerical Features:
- age
- fnlgt
- education-num
- capital-gain
- capital-loss
- hours-per-week

---

## Evaluation Data

The evaluation data was the **20% test split** from the original dataset. The same preprocessing was applied to the test data to ensure compatibility with the trained model.

---

## Metrics

The model was evaluated using the following metrics:
- **Precision**
- **Recall**
- **F1-score**

### Performance:
- Precision: 0.7383
- Recall: 0.6077
- F1-score: 0.6667
The model was tested on slices of the data based on categorical features to understand how performance varies for each subgroup.

---

## Ethical Considerations

- **Bias and Fairness:** The model may exhibit biases present in the original Census dataset, particularly concerning race, sex, and native country. While the model aims to generalize well, the inherent biases in the data can propagate through predictions.
- **Privacy:** The dataset contains socio-economic information, and therefore the model predictions should be handled with caution, particularly when deployed in public environments.
- **Misuse Risk:** Predictions from this model should not be used as the sole criterion for decision-making, especially in financial, hiring, or other sensitive applications.

---

## Caveats and Recommendations

- The model's performance may degrade if applied to datasets with significantly different distributions than the training data.
- Periodic retraining and evaluation should be performed to ensure the model remains relevant and accurate.
- Users are encouraged to monitor model predictions and assess bias periodically to avoid perpetuating harmful stereotypes or unfair practices.