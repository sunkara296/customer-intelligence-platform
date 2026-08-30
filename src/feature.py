def add_average_order_value(df):
    df['average_order_value'] = df['total_spend'] / df['total_orders']
    df['average_order_value'] = df['average_order_value'].fillna(0)

    return df