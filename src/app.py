from data_loader import load_data
from feature import build_customer_features

customers_df = load_data("src/data/customers.csv")
orders_df = load_data("src/data/orders.csv")

customer_features_df = build_customer_features(
    customers_df,
    orders_df,
    '2026-07-01'
)

print(customer_features_df)
