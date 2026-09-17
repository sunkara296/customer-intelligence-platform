import pandas as pd

def add_average_order_value(df):
    df['average_order_value'] = df['total_spend'] / df['total_orders']
    df['average_order_value'] = df['average_order_value'].fillna(0)

    return df


def build_customer_features(customers_df, orders_df, snapshot_date):
    snapshot_date = pd.Timestamp(snapshot_date)

    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    customer_order_summary = (
    orders_df.groupby('customer_id')
    .agg(
        total_orders=('order_id', 'count'),
        total_spend=('order_amount', 'sum'),
        last_order_date=('order_date', 'max')
    )
    .reset_index()
    )

    customer_features_df=customers_df.merge(customer_order_summary,on='customer_id',how='left')


    customer_features_df['total_orders']=customer_features_df['total_orders'].fillna(0).astype(int)

    customer_features_df['total_spend']=customer_features_df['total_spend'].fillna(0)


    customer_features_df = add_average_order_value(customer_features_df)


    customer_features_df['recency_days'] = (
    snapshot_date - customer_features_df['last_order_date']
    ).dt.days


    customer_features_df['signup_date'] = pd.to_datetime(
    customer_features_df['signup_date']
    )

    customer_features_df['tenure_days'] = (
    snapshot_date - customer_features_df['signup_date']
    ).dt.days

    return customer_features_df