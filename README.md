# California Housing Regression Analysis

This project demonstrates a complete, reproducible regression-analysis workflow using the California Housing dataset from `scikit-learn`.

The project is designed to work well with OpenAI Codex or other coding agents. The repository includes an `AGENTS.md` file that defines project-specific instructions, expected commands, validation rules, and completion criteria.

## Objective

Predict `MedHouseVal`, the median house value for California census block groups, using housing, demographic, and geographic predictors.

The target is measured in units of `$100,000`.

Examples:

| Target Value | Approximate Dollar Value |
|---:|---:|
| 2.5 | $250,000 |
| 4.0 | $400,000 |

## Dataset

The dataset is loaded with:

```python
from sklearn.datasets import fetch_california_housing
```

It includes 20,640 rows and 8 predictor columns.

| Feature | Meaning |
|---|---|
| `MedInc` | Median income in the block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

Target:

| Target | Meaning |
|---|---|
| `MedHouseVal` | Median house value in units of $100,000 |

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── california_regression.py
├── tests/
│   ├── __init__.py
│   └── test_california_regression.py
├── .gitignore
├── AGENTS.md
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

## Models Compared

The analysis compares:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regression

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Square root of MSE; interpretable in target units |
| R² | Proportion of variance explained by the model |

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Run the Analysis

```bash
python src/california_regression.py
```

The script generates CSV, JSON, and PNG outputs in the `outputs/` directory.

## Run Tests

```bash
pytest
```

## Expected Outputs

After running the analysis, the following files should exist:

```text
outputs/
├── analysis_summary.json
├── averooms_vs_target.png
├── correlation_heatmap.png
├── cross_validation_results.csv
├── houseage_vs_target.png
├── latitude_vs_target.png
├── linear_coefficients.csv
├── longitude_vs_target.png
├── medinc_vs_target.png
├── model_comparison.csv
├── predicted_vs_actual.png
└── residual_plot.png
```

## Interpretation Guidance

Typical results show that:

- Random Forest usually outperforms the linear models because housing prices are nonlinear and geographically clustered.
- `MedInc` is typically the strongest positive predictor in the linear model.
- `Latitude` and `Longitude` are important but should be interpreted carefully because they encode geographic location rather than direct causal effects.
- RMSE should be translated into approximate dollar terms by multiplying by 100,000.

Example:

```text
RMSE = 0.50
Approximate error = $50,000
```

## GitHub Usage

Recommended first commit:

```bash
git init
git add .
git commit -m "Initial California Housing regression project"
```

Optional GitHub push:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/california-regression-codex-project.git
git push -u origin main
```

## Notes for Codex

See `AGENTS.md` for project-specific coding-agent instructions.
