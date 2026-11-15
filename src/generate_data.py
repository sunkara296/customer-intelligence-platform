import pandas as pd
import numpy as np

np.random.seed(42)

NUM_CUSTOMERS = 1000
SNAPSHOT_DATE = pd.Timestamp("2026-07-01")

# --------------------------------------------------
# 1. Generate customers
# --------------------------------------------------

customer_ids = np.arange(1001, 1001 + NUM_CUSTOMERS)

start_date = pd.Timestamp("2024-01-01")

random_signup_days = np.random.randint(
    0,
    (SNAPSHOT_DATE - start_date).days,
    size=NUM_CUSTOMERS
)

signup_dates = start_date + pd.to_timedelta(
    random_signup_days,
    unit="D"
)

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "signup_date": signup_dates
})


# --------------------------------------------------
# 2. Generate historical orders
# --------------------------------------------------

orders = []
order_id = 10001

for customer_id, signup_date in zip(customer_ids, signup_dates):

    # Customers have different levels of engagement
    engagement = np.random.beta(2, 2)

    # More engaged customers generally place more orders
    num_orders = np.random.poisson(
        2 + engagement * 12
    )

    for _ in range(num_orders):

        available_days = (SNAPSHOT_DATE - signup_date).days

        if available_days <= 0:
            continue

        # More engaged customers tend to have more recent orders
        days_ago = int(
            np.random.exponential(
                scale=20 + (1 - engagement) * 100
            )
        )

        days_ago = min(days_ago, available_days)

        order_date = SNAPSHOT_DATE - pd.Timedelta(days=days_ago)

        order_amount = round(
            np.random.lognormal(mean=4.5, sigma=0.6),
            2
        )

        orders.append([
            order_id,
            customer_id,
            order_date,
            order_amount
        ])

        order_id += 1


# --------------------------------------------------
# 3. Generate future 30-day behavior
# --------------------------------------------------

for customer_id in customer_ids:

    customer_history = [
        row for row in orders
        if row[1] == customer_id
        and row[2] <= SNAPSHOT_DATE
    ]

    if customer_history:

        last_order_date = max(
            row[2] for row in customer_history
        )

        recency_days = (
            SNAPSHOT_DATE - last_order_date
        ).days

        historical_order_count = len(customer_history)

    else:
        recency_days = 365
        historical_order_count = 0

    # Probability of purchasing again.
    # Recent/frequent customers are more likely to return.
    purchase_score = (
        1.5
        - 0.025 * recency_days
        + 0.12 * historical_order_count
    )

    purchase_probability = (
        1 / (1 + np.exp(-purchase_score))
    )

    has_future_order = (
        np.random.random() < purchase_probability
    )

    if has_future_order:

        num_future_orders = np.random.randint(1, 4)

        for _ in range(num_future_orders):

            days_after_snapshot = np.random.randint(1, 31)

            order_date = (
                SNAPSHOT_DATE
                + pd.Timedelta(days=days_after_snapshot)
            )

            order_amount = round(
                np.random.lognormal(mean=4.5, sigma=0.6),
                2
            )

            orders.append([
                order_id,
                customer_id,
                order_date,
                order_amount
            ])

            order_id += 1


# --------------------------------------------------
# 4. Save raw datasets
# --------------------------------------------------

orders_df = pd.DataFrame(
    orders,
    columns=[
        "order_id",
        "customer_id",
        "order_date",
        "order_amount"
    ]
)

customers_df.to_csv(
    "src/data/customers.csv",
    index=False
)

orders_df.to_csv(
    "src/data/orders.csv",
    index=False
)

print("Customers:", len(customers_df))
print("Orders:", len(orders_df))
print()
print(customers_df.head())
print()
print(orders_df.head())