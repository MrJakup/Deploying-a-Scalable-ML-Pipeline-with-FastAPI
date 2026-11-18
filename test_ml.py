import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    train_model,
)


@pytest.fixture
def sample_data():
    """Creates a small synthetic dataset matching the census schema."""
    data = pd.DataFrame(
        {
            "age": [39, 50, 38, 53, 28, 37, 49, 52, 31, 42],
            "workclass": [
                "State-gov",
                "Self-emp-not-inc",
                "Private",
                "Private",
                "Private",
                "Private",
                "Private",
                "Self-emp-not-inc",
                "Private",
                "Private",
            ],
            "education": [
                "Bachelors",
                "Bachelors",
                "HS-grad",
                "11th",
                "Bachelors",
                "Masters",
                "9th",
                "HS-grad",
                "Masters",
                "Bachelors",
            ],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-spouse-absent",
                "Married-civ-spouse",
                "Never-married",
                "Married-civ-spouse",
            ],
            "occupation": [
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
                "Handlers-cleaners",
                "Prof-specialty",
                "Exec-managerial",
                "Other-service",
                "Exec-managerial",
                "Prof-specialty",
                "Exec-managerial",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
                "Wife",
                "Wife",
                "Not-in-family",
                "Husband",
                "Not-in-family",
                "Husband",
            ],
            "race": [
                "White",
                "White",
                "White",
                "Black",
                "Black",
                "White",
                "Black",
                "White",
                "White",
                "White",
            ],
            "sex": [
                "Male",
                "Male",
                "Male",
                "Male",
                "Female",
                "Female",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "capital-gain": [2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178],
            "capital-loss": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "hours-per-week": [40, 13, 40, 40, 40, 40, 16, 45, 50, 40],
            "native-country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "Cuba",
                "United-States",
                "Jamaica",
                "United-States",
                "United-States",
                "United-States",
            ],
            "salary": [
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
                ">50K",
            ],
        }
    )
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
    return data, cat_features


def test_train_model(sample_data):
    """Tests that train_model returns a fitted RandomForestClassifier."""
    data, cat_features = sample_data
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")


def test_inference(sample_data):
    """Tests that inference returns binary predictions matching input length."""
    data, cat_features = sample_data
    X, y, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_performance_on_categorical_slice(sample_data):
    """Tests that slice evaluation returns valid precision, recall, and fbeta floats."""
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)

    precision, recall, fbeta = performance_on_categorical_slice(
        data=data,
        column_name="sex",
        slice_value="Male",
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
    )

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
