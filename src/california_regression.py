"""California Housing regression analysis.

This module loads the California Housing dataset, performs exploratory data
analysis, trains several regression models, evaluates them, and writes output
artifacts to the outputs directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TARGET_COLUMN = "MedHouseVal"


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset as a pandas DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame.copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into feature matrix X and target vector y."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_models() -> dict[str, Any]:
    """Create the regression models used in the comparison."""
    return {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, float | str], np.ndarray]:
    """Evaluate a fitted model on the test set."""
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "model": name,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
    }, predictions


def save_eda_plots(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Save exploratory analysis plots."""
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("California Housing Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

    sampled_df = df.sample(min(3000, len(df)), random_state=RANDOM_STATE)

    for feature in ["MedInc", "AveRooms", "HouseAge", "Latitude", "Longitude"]:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=sampled_df,
            x=feature,
            y=TARGET_COLUMN,
            alpha=0.4,
        )
        plt.title(f"{feature} vs Median House Value")
        plt.xlabel(feature)
        plt.ylabel("Median House Value ($100,000s)")
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature.lower()}_vs_target.png")
        plt.close()


def save_diagnostic_plots(
    y_test: pd.Series,
    predictions: np.ndarray,
    model_name: str,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Save predicted-vs-actual and residual diagnostic plots."""
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.35)
    plt.xlabel("Actual Median House Value")
    plt.ylabel("Predicted Median House Value")
    plt.title(f"Predicted vs Actual: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual.png")
    plt.close()

    residuals = y_test - predictions

    plt.figure(figsize=(7, 5))
    plt.scatter(predictions, residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Median House Value")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot: {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_plot.png")
    plt.close()


def tune_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune Ridge Regression alpha with grid search."""
    ridge_grid = GridSearchCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    ridge_grid.fit(X_train, y_train)
    return ridge_grid


def save_linear_coefficients(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Train a linear model and save its coefficients."""
    linear_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    linear_model.fit(X_train, y_train)

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear_model.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", ascending=False)

    coefficients.to_csv(output_dir / "linear_coefficients.csv", index=False)
    return coefficients


def run_analysis(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run the full regression analysis and return a summary dictionary."""
    output_dir.mkdir(exist_ok=True)

    df = load_data()

    print("Dataset shape:", df.shape)
    print("\nFirst rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nSummary statistics:")
    print(df.describe())

    save_eda_plots(df, output_dir)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    results: list[dict[str, float | str]] = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        result, predictions = evaluate_model(name, model, X_test, y_test)
        results.append(result)
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\nModel comparison:")
    print(results_df)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)

    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        cv_results.append(
            {
                "model": name,
                "cv_rmse_mean": float(-scores.mean()),
                "cv_rmse_std": float(scores.std()),
            }
        )

    cv_df = pd.DataFrame(cv_results).sort_values("cv_rmse_mean")
    print("\nCross-validation results:")
    print(cv_df)
    cv_df.to_csv(output_dir / "cross_validation_results.csv", index=False)

    ridge_grid = tune_ridge(X_train, y_train)
    tuned_result, _ = evaluate_model(
        "Tuned Ridge Regression",
        ridge_grid.best_estimator_,
        X_test,
        y_test,
    )

    print("\nBest Ridge alpha:", ridge_grid.best_params_)
    print("Tuned Ridge result:", tuned_result)

    best_model_name = str(results_df.iloc[0]["model"])
    best_predictions = predictions_by_model[best_model_name]

    save_diagnostic_plots(y_test, best_predictions, best_model_name, output_dir)

    coefficients = save_linear_coefficients(
        X_train,
        y_train,
        list(X.columns),
        output_dir,
    )

    print("\nLinear regression coefficients:")
    print(coefficients)

    summary = {
        "dataset_shape": list(df.shape),
        "target": TARGET_COLUMN,
        "test_results": results,
        "cross_validation_results": cv_results,
        "best_test_model": best_model_name,
        "best_ridge_params": ridge_grid.best_params_,
        "tuned_ridge_result": tuned_result,
        "notes": [
            "MedInc is usually the strongest positive linear predictor.",
            "Latitude and Longitude often capture important geographic effects.",
            "Random Forest commonly outperforms linear models because housing prices are nonlinear and geographically clustered.",
            "Target values are expressed in units of $100,000.",
        ],
    }

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def main() -> None:
    """Command-line entry point."""
    run_analysis()


if __name__ == "__main__":
    main()
