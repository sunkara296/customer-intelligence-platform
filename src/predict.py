"""Verify that the persisted churn model can be loaded and used."""
from pathlib import Path

import joblib

from train_model import DATA_DIR, ROOT, SNAPSHOT_DATE
from data_loader import load_data
from feature import build_customer_features


def main():
    artifact = joblib.load(ROOT / 'models' / 'churn_model.joblib')
    customers = load_data(DATA_DIR / 'customers.csv')
    orders = load_data(DATA_DIR / 'orders.csv')
    features = build_customer_features(customers.copy(), orders.copy(), SNAPSHOT_DATE)
    sample = features.iloc[[0]]
    probability = float(artifact['model'].predict_proba(
        sample[artifact['feature_columns']]
    )[0, 1])
    prediction = int(probability >= artifact['threshold'])
    print('Customer ID:', sample.iloc[0]['customer_id'])
    print('Churn probability:', round(probability, 4))
    print('Threshold:', round(artifact['threshold'], 4))
    print('Predicted churn:', prediction)


if __name__ == '__main__':
    main()
