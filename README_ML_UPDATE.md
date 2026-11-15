# Customer Intelligence Platform — ML completion

## Training

From the repository root on Windows:

```cmd
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\train_model.py
python src\predict.py
```

Requires existing `src/data/customers.csv`, `src/data/orders.csv`, `src/data_loader.py`, and `src/feature.py`.

## What the pipeline does

- Builds customer-level snapshot-safe features as of 2026-07-01.
- Labels churn as zero orders during the following 30 days.
- Uses a stratified 70/15/15 train/validation/test split.
- Compares balanced Logistic Regression and Random Forest using five-fold CV on training rows only.
- Selects the model with higher mean cross-validated Average Precision, tie-broken by ROC-AUC.
- Chooses a validation-set threshold to maximize recall with a minimum precision of 25%, falling back to best F1 if none qualify.
- Refits the selected model on train + validation rows, evaluates once on the test split, and writes the model plus evaluation report.
- Verifies that the saved model can be reloaded for inference.

Outputs: `models/churn_model.joblib`, `models/evaluation.json`.

## Interpretation and limitations

Logistic Regression coefficients refer to standardized input features. Random Forest impurity-based importances are not causal and can favor some features. All evaluation uses synthetic data. Earlier development already inspected this test split, so the result is an illustrative holdout, not an unbiased final estimate. Use a new out-of-time holdout and real data before production claims.

Never load untrusted `.joblib` files: joblib deserialization can execute code.
