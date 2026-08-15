from customer import Customer

customer1 = Customer(
    customer_id=101,
    total_orders=5,
    total_spend=1000
)

print(customer1.average_order_value())
