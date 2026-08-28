"""Generate a synthetic retail SQLite database for the demo and evals.

Deterministic (seeded) so the database — and therefore the evaluation results —
are reproducible for anyone who clones the repo.
"""

from __future__ import annotations

import os
import random
import sqlite3
from bisect import bisect_right
from datetime import date, timedelta

SEED = 42
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {
    "Electronics": [("Wireless Mouse", 29.99), ("USB-C Hub", 49.99), ("Webcam", 79.99)],
    "Home": [("Desk Lamp", 39.99), ("Coffee Mug", 12.99), ("Throw Blanket", 34.99)],
    "Office": [("Notebook", 8.99), ("Pen Set", 15.99), ("Standing Desk Mat", 59.99)],
    "Fitness": [("Yoga Mat", 24.99), ("Resistance Bands", 19.99), ("Water Bottle", 17.99)],
}

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT NOT NULL,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

FIRST = ["Ava", "Liam", "Noah", "Emma", "Olivia", "Mia", "Ethan", "Sofia", "Lucas", "Aria"]
LAST = ["Chen", "Patel", "Garcia", "Smith", "Kim", "Nguyen", "Brown", "Lopez", "Khan", "Reed"]


def build(db_path: str = DB_PATH) -> None:
    rng = random.Random(SEED)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    # products
    products: list[tuple[int, str, str, float]] = []
    pid = 1
    for category, items in CATEGORIES.items():
        for name, price in items:
            products.append((pid, name, category, price))
            pid += 1
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    # customers: 120, signing up across 2023-2024
    start = date(2023, 1, 1)
    customers = []
    for cid in range(1, 121):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        region = rng.choice(REGIONS)
        signup = start + timedelta(days=rng.randint(0, 700))
        customers.append((cid, name, region, signup.isoformat()))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    # An order may only be attributed to a customer who had already signed up
    # on the day it was placed. Drawing the customer uniformly from all 120 --
    # as this generator used to -- put 42% of first orders *before* the
    # customer's own signup_date, which makes any signup-relative metric
    # (time to first order, activation rate) meaningless.
    #
    # The constraint is applied to the customer, not the date: order dates stay
    # uniform across 2024, and a late-2024 signup simply has fewer days on
    # which it could have ordered. Clamping the date instead would have bunched
    # orders toward year-end and distorted every monthly series in the catalog.
    signups = sorted((date.fromisoformat(row[3]), row[0]) for row in customers)
    signup_days = [day for day, _ in signups]

    # orders + items across 2024
    order_id = 1
    item_id = 1
    orders, items = [], []
    for _ in range(900):
        order_day = date(2024, 1, 1) + timedelta(days=rng.randint(0, 364))
        # bisect_right gives the number of customers signed up on or before
        # order_day; that prefix of `signups` is the eligible pool.
        eligible = bisect_right(signup_days, order_day)
        cid = signups[rng.randrange(eligible)][1]
        orders.append((order_id, cid, order_day.isoformat()))
        for _ in range(rng.randint(1, 4)):
            prod = rng.choice(products)
            qty = rng.randint(1, 3)
            items.append((item_id, order_id, prod[0], qty, prod[3]))
            item_id += 1
        order_id += 1
    conn.executemany("INSERT INTO orders VALUES (?,?,?)", orders)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    conn.commit()
    conn.close()
    print(
        f"Built {db_path}: {len(customers)} customers, {len(products)} products, "
        f"{len(orders)} orders, {len(items)} order items."
    )


if __name__ == "__main__":
    build()
