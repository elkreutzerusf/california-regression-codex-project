# AGENTS.md — California Housing Regression Project

This file provides guidance for AI coding agents, including OpenAI Codex, working in this repository.

It defines:
- how to navigate the project
- how to run and validate work
- coding and modeling conventions
- expected outputs
- completion criteria

---

# 1. Project Overview

This project performs a complete regression analysis using the California Housing dataset.

Primary objectives:
- Predict `MedHouseVal`
- Compare multiple regression models
- Produce evaluation metrics
- Generate visual diagnostics
- Save reproducible project outputs

Target variable:
- `MedHouseVal`

Target units:
- `$100,000`

---

# 2. Project Structure

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

Agent rules:
- Primary application logic belongs in `src/`.
- Tests belong in `tests/`.
- Generated files belong in `outputs/`.
- Do not manually edit generated output files unless explicitly instructed.
- Do not introduce new dependencies unless required by the task.

---

# 3. Environment & Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

---

# 4. Execution Commands

Run the full regression pipeline:

```bash
python src/california_regression.py
```

Run tests:

```bash
pytest
```

Verify generated outputs:

```bash
ls outputs/
```

---

# 5. Expected Outputs

After running the main script, the following files should exist:

- `outputs/model_comparison.csv`
- `outputs/cross_validation_results.csv`
- `outputs/linear_coefficients.csv`
- `outputs/analysis_summary.json`
- `outputs/correlation_heatmap.png`
- `outputs/medinc_vs_target.png`
- `outputs/averooms_vs_target.png`
- `outputs/houseage_vs_target.png`
- `outputs/latitude_vs_target.png`
- `outputs/longitude_vs_target.png`
- `outputs/predicted_vs_actual.png`
- `outputs/residual_plot.png`

If any required output is missing after script execution, the task is incomplete.

---

# 6. Modeling Requirements

The pipeline must include:

## Dataset
- Use `sklearn.datasets.fetch_california_housing(as_frame=True)`.

## Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regression

## Evaluation Metrics
- MAE
- MSE
- RMSE
- R²

## Validation
- Train/test split using `test_size=0.20`
- `random_state=42`
- 5-fold cross-validation

## Hyperparameter Tuning
- Tune Ridge `alpha` using `GridSearchCV`.

---

# 7. Coding Conventions

General:
- Python 3.9+
- Use clear variable names
- Prefer readability over cleverness
- Avoid hidden side effects
- Keep functions small and single-purpose

Data science:
- Use pandas for tabular data manipulation
- Use scikit-learn pipelines where appropriate
- Keep preprocessing explicit
- Make train/test split reproducible

Project conventions:
- The main executable script is `src/california_regression.py`.
- Keep generated charts, CSVs, and JSON files in `outputs/`.
- Avoid absolute paths.
- Use `pathlib.Path` for filesystem paths.

---

# 8. Visualization Standards

Use:
- matplotlib
- seaborn

Rules:
- Save plots to `outputs/`.
- Do not require interactive plot display.
- Use meaningful titles and axis labels.
- Keep visualizations readable for a beginner regression-analysis audience.

Required plots:
- Correlation heatmap
- Feature-vs-target scatterplots
- Predicted-vs-actual plot
- Residual plot

---

# 9. Interpretation Guidelines

When explaining results:

- Explain model performance in plain language.
- Convert RMSE into approximate dollars:
  - `RMSE * 100,000`
- Highlight likely strongest predictors:
  - `MedInc`
  - `Latitude`
  - `Longitude`
- Explain that geographic variables are predictive but should not be interpreted as simple causal effects.
- Explain why Random Forest often performs better:
  - nonlinear relationships
  - feature interactions
  - geographic clustering

---

# 10. Validation Checklist

Before marking any task complete, verify:

- [ ] Main script runs without errors.
- [ ] Tests pass with `pytest`.
- [ ] Required output files are generated.
- [ ] Model comparison is printed.
- [ ] Cross-validation results are printed.
- [ ] README remains consistent with project behavior.
- [ ] No unused imports were introduced.
- [ ] No hardcoded local machine paths were introduced.
- [ ] No unnecessary dependencies were added.

---

# 11. Safe Execution Rules

Ask before:
- installing packages
- running shell commands
- deleting files
- changing multiple unrelated files

Never:
- delete files without explicit instruction
- commit secrets
- add API keys
- add credentials
- write machine-specific absolute paths
- introduce heavyweight dependencies unnecessarily

---

# 12. Iteration Strategy

For multi-step tasks:

1. Inspect the current repository.
2. Plan changes.
3. Modify the smallest necessary set of files.
4. Run tests.
5. Run the analysis if relevant.
6. Inspect outputs.
7. Summarize changes and results.

Do not stop after code generation when execution and validation are possible.

---

# 13. Completion Criteria

A task is complete only when:

- Code executes successfully.
- Tests pass.
- Required outputs are generated.
- Results are validated.
- Any changed behavior is documented.

---

# 14. Optional Enhancements

Only implement these if explicitly requested:

- Permutation importance
- SHAP explanations
- Gradient Boosting
- XGBoost or LightGBM
- expanded hyperparameter tuning
- model persistence with joblib
- notebook conversion
- HTML report generation
- slide deck generation

---

# 15. Agent Behavior Expectations

- Be deterministic.
- Be concise but complete.
- Prefer minimal correct changes.
- Validate empirically.
- Explain assumptions.
- Do not speculate when results can be measured.

---

# End of AGENTS.md
