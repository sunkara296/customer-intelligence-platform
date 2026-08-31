from data_loader import load_customer_data

from feature import add_average_order_value

df = load_customer_data("src/data/customers.csv")

df = add_average_order_value(df)

print(df)
