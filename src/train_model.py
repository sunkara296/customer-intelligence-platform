"""Train, compare, evaluate, and persist the customer churn classifier."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from feature import build_customer_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'src' / 'data'
MODEL_DIR = ROOT / 'models'
SNAPSHOT_DATE = '2026-07-01'
HORIZON_DAYS = 30
FEATURE_COLUMNS = [
    'total_orders', 'total_spend', 'average_order_value',
    'recency_days', 'tenure_days', 'has_ordered',
]
RANDOM_STATE = 42


def build_churn_labels(customers_df: pd.DataFrame, orders_df: pd.DataFrame,
                       snapshot_date: str = SNAPSHOT_DATE) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot_date)
    orders = orders_df.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    future = orders.loc[
        (orders['order_date'] > snapshot)
        & (orders['order_date'] <= snapshot + pd.Timedelta(days=HORIZON_DAYS))
    ]
    counts = future.groupby('customer_id').size().rename('future_order_count')
    labels = customers_df[['customer_id']].merge(
        counts.reset_index(), on='customer_id', how='left', validate='one_to_one'
    )
    labels['future_order_count'] = labels['future_order_count'].fillna(0).astype(int)
    labels['churn'] = (labels['future_order_count'] == 0).astype(int)
    return labels


def prepare_dataset() -> pd.DataFrame:
    customers = load_data(DATA_DIR / 'customers.csv')
    orders = load_data(DATA_DIR / 'orders.csv')
    if customers['customer_id'].duplicated().any():
        raise ValueError('Customer IDs must be unique')
    features = build_customer_features(customers.copy(), orders.copy(), SNAPSHOT_DATE)
    labels = build_churn_labels(customers, orders, SNAPSHOT_DATE)
    dataset = features.merge(
        labels[['customer_id', 'churn']], on='customer_id', how='left',
        validate='one_to_one'
    )
    if dataset[FEATURE_COLUMNS + ['churn']].isna().any().any():
        raise ValueError('Missing ML features or labels; check feature engineering')
    if not np.isfinite(dataset[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite values in ML features')
    if dataset['churn'].nunique() < 2:
        raise ValueError('Both churn classes are required for training')
    return dataset


def candidates():
    return {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
            )),
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, scores)),
        'average_precision': float(average_precision_score(y_true, scores)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, preds).tolist(),
    }


def main():
    data = prepare_dataset()
    X, y = data[FEATURE_COLUMNS], data['churn']
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print('Rows:', len(data), '| Churn prevalence:', round(float(y.mean()), 3))
    print('Train/validation/test:', len(X_train), len(X_val), len(X_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, estimator in candidates().items():
        scores = cross_validate(
            estimator, X_train, y_train, cv=cv,
            scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision'},
            n_jobs=1,
        )
        cv_results[name] = {
            'roc_auc_mean': float(scores['test_roc_auc'].mean()),
            'roc_auc_std': float(scores['test_roc_auc'].std()),
            'average_precision_mean': float(scores['test_average_precision'].mean()),
            'average_precision_std': float(scores['test_average_precision'].std()),
        }
        print(f"{name}: ROC-AUC={cv_results[name]['roc_auc_mean']:.3f} "
              f"(+/-{cv_results[name]['roc_auc_std']:.3f}), "
              f"AP={cv_results[name]['average_precision_mean']:.3f} "
              f"(+/-{cv_results[name]['average_precision_std']:.3f})")

    # Select model by cross-validated average precision (positive class is rare).
    winner = max(cv_results, key=lambda name: (
        cv_results[name]['average_precision_mean'],
        cv_results[name]['roc_auc_mean'],
    ))
    model = candidates()[winner]
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]

    # Choose threshold on validation only; maximize recall subject to >=25% precision.
    # If no threshold qualifies, choose the best F1. Threshold candidates include
    # all distinct observed validation scores plus a value above the maximum.
    thresholds = np.unique(np.r_[0.0, val_scores, np.nextafter(1.0, 2.0)])
    choices = []
    for threshold in thresholds:
        preds = (val_scores >= threshold).astype(int)
        choices.append({
            'threshold': float(threshold),
            'precision': float(precision_score(y_val, preds, zero_division=0)),
            'recall': float(recall_score(y_val, preds, zero_division=0)),
            'f1': float(f1_score(y_val, preds, zero_division=0)),
        })
    qualified = [x for x in choices if x['precision'] >= 0.25 and x['recall'] > 0]
    if qualified:
        selected = max(qualified, key=lambda x: (x['recall'], x['precision'], x['f1']))
        threshold_rule = 'maximize validation recall with precision >= 0.25'
    else:
        selected = max(choices, key=lambda x: (x['f1'], x['recall']))
        threshold_rule = 'maximize validation F1 (precision floor infeasible)'
    threshold = selected['threshold']
    print('Selected model:', winner)
    print('Validation threshold:', round(threshold, 4), '|', threshold_rule)
    print('Validation metrics:', json.dumps(metrics(y_val, val_scores, threshold), indent=2))

    # Refit selected configuration on development data only, never on test data.
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    final_model = candidates()[winner]
    final_model.fit(X_dev, y_dev)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, test_scores, threshold)
    print('Test metrics:', json.dumps(test_metrics, indent=2))
    print('Test classification report:')
    print(classification_report(
        y_test, (test_scores >= threshold).astype(int), zero_division=0
    ))

    if winner == 'logistic_regression':
        coefficients = final_model.named_steps['classifier'].coef_[0]
        interpretation = sorted(
            zip(FEATURE_COLUMNS, coefficients), key=lambda item: abs(item[1]),
            reverse=True,
        )
        print('Standardized LR coefficients (association, not causation):')
        for feature, coefficient in interpretation:
            print(f'  {feature:24s} {coefficient:+.4f}')
    else:
        interpretation = sorted(
            zip(FEATURE_COLUMNS, final_model.feature_importances_),
            key=lambda item: item[1], reverse=True,
        )
        print('RF impurity-based importances (can be biased):')
        for feature, importance in interpretation:
            print(f'  {feature:24s} {importance:.4f}')

    MODEL_DIR.mkdir(exist_ok=True)
    artifact = {
        'model': final_model,
        'feature_columns': FEATURE_COLUMNS,
        'threshold': threshold,
        'snapshot_date': SNAPSHOT_DATE,
        'horizon_days': HORIZON_DAYS,
        'model_name': winner,
    }
    artifact_path = MODEL_DIR / 'churn_model.joblib'
    joblib.dump(artifact, artifact_path)
    report = {
        'data_rows': len(data),
        'churn_prevalence': float(y.mean()),
        'splits': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
        'cross_validation': cv_results,
        'selected_model': winner,
        'threshold': threshold,
        'threshold_rule': threshold_rule,
        'validation_metrics': metrics(y_val, val_scores, threshold),
        'test_metrics': test_metrics,
        'feature_interpretation': [
            {'feature': name, 'value': float(value)} for name, value in interpretation
        ],
        'caveats': [
            'Synthetic dataset: metrics do not establish real-world performance.',
            'This test split was previously inspected during development; treat it as illustrative, not pristine.',
            'A new time-based external holdout is needed for unbiased deployment claims.',
            'Validation-derived threshold is not a calibrated business cost optimum.',
        ],
    }
    (MODEL_DIR / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved:', artifact_path)
    print('Saved:', MODEL_DIR / 'evaluation.json')


if __name__ == '__main__':
    main()
