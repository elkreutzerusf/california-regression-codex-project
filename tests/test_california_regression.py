"""Unit tests for the California Housing regression analysis."""

from pathlib import Path

import pandas as pd

from src.california_regression import (
    TARGET_COLUMN,
    build_models,
    evaluate_model,
    load_data,
    run_analysis,
    split_features_target,
)


class ConstantModel:
    """Simple model used to test evaluation metrics."""

    def predict(self, X):
        return [1.0 for _ in range(len(X))]


def test_load_data_contains_expected_target():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert TARGET_COLUMN in df.columns
    assert len(df) > 0


def test_split_features_target_removes_target():
    df = load_data()
    X, y = split_features_target(df)

    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)


def test_build_models_contains_required_models():
    models = build_models()

    assert "Linear Regression" in models
    assert "Ridge Regression" in models
    assert "Lasso Regression" in models
    assert "Random Forest" in models


def test_evaluate_model_returns_expected_metrics():
    X_test = pd.DataFrame({"feature": [1, 2, 3]})
    y_test = pd.Series([1.0, 2.0, 3.0])

    result, predictions = evaluate_model("Constant", ConstantModel(), X_test, y_test)

    assert result["model"] == "Constant"
    assert "mae" in result
    assert "mse" in result
    assert "rmse" in result
    assert "r2" in result
    assert len(predictions) == len(y_test)


def test_run_analysis_creates_core_outputs(tmp_path: Path):
    summary = run_analysis(output_dir=tmp_path)

    required_files = [
        "model_comparison.csv",
        "cross_validation_results.csv",
        "linear_coefficients.csv",
        "analysis_summary.json",
        "correlation_heatmap.png",
        "predicted_vs_actual.png",
        "residual_plot.png",
    ]

    for file_name in required_files:
        assert (tmp_path / file_name).exists()

    assert summary["target"] == TARGET_COLUMN
    assert "best_test_model" in summary
