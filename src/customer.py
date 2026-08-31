class Customer:
    def __init__(self, customer_id, total_orders, total_spend):
        self.customer_id = customer_id
        self.total_orders = total_orders
        self.total_spend = total_spend

    def average_order_value(self):
        if self.total_orders == 0:
            return 0

        return self.total_spend / self.total_orders
    