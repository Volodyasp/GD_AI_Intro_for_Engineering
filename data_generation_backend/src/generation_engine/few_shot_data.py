FEW_SHOT_EXAMPLES = [
    {
        "question": "How many restaurants are there in each city?",
        "sql_query": "SELECT city, COUNT(*) AS restaurant_count FROM restaurants GROUP BY city ORDER BY restaurant_count DESC;"
    },
    {
        "question": "Which customers placed the most orders?",
        "sql_query": """
            SELECT c.name, COUNT(o.order_id) AS total_orders 
            FROM customers c 
            JOIN orders o ON c.customer_id = o.customer_id 
            GROUP BY c.customer_id, c.name 
            ORDER BY total_orders DESC 
            LIMIT 5;
        """
    },
    {
        "question": "Show me the top 3 most expensive menu items.",
        "sql_query": "SELECT name, price, description FROM menu ORDER BY price DESC LIMIT 3;"
    },
    {
        "question": "List all orders placed by 'Alice Smith' including the date.",
        "sql_query": """
            SELECT o.order_id, o.order_date, o.total_amount 
            FROM orders o 
            JOIN customers c ON o.customer_id = c.customer_id 
            WHERE c.name = 'Alice Smith';
        """
    },
    {
        "question": "What is the average rating of Italian restaurants?",
        "sql_query": "SELECT AVG(rating) as avg_rating FROM restaurants WHERE cuisine_type = 'Italian';"
    }
]
