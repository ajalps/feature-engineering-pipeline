# Feature Engineering & Preprocessing Pipeline — Credit Default Risk

A leak-free `scikit-learn` preprocessing pipeline: feature encoding, scaling, missing-value
imputation, and feature-importance / correlation analysis, all wrapped in a reusable
`Pipeline` + `ColumnTransformer`.

## Contents

| File | Purpose |
|---|---|
| [`feature_engineering_pipeline.ipynb`](feature_engineering_pipeline.ipynb) | The full pipeline: load data, split, preprocess, fit, feature importance |
| `requirements.txt` | Python dependencies |

## Approach

- **Dataset:** synthetic, seeded stand-in for a credit-default-risk dataset (numeric
  features, a low-cardinality categorical, a high-cardinality categorical, and injected
  missing values) — no real dataset was supplied for this exercise. Swap the loading
  cell for a real CSV to use it on production data.
- **Leakage prevention:** `train_test_split` runs first, before any imputer, scaler, or
  encoder is fit. All preprocessing lives inside a `Pipeline` that is fit only on the
  training split.
- **Encoding:** `OneHotEncoder` for the low-cardinality categorical column,
  `TargetEncoder` (scikit-learn's built-in, cross-fitted version) for the
  high-cardinality one, to avoid a huge sparse one-hot expansion.
- **Imputation + scaling:** median imputation + `StandardScaler` for numeric columns;
  most-frequent imputation for categoricals.
- **Feature importance:** tree-based importance from a `RandomForestClassifier`, cross-checked
  against mutual information (`mutual_info_classif`) computed on the same transformed
  training features.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook feature_engineering_pipeline.ipynb
```

Or open it directly in Google Colab / VS Code / GitHub's notebook viewer.
