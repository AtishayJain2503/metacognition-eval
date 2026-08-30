"""
nemo_eval.datasets.synthetic
============================
Hermetic offline synthetic dataset and database fixture generator for 100%
deterministic CI/CD evaluation environments.
"""

from datetime import datetime, timedelta
import math
import os
import random
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from nemo_eval.datasets.base import BenchmarkTask


class SyntheticBenchmarkGenerator:
    """
    Generator for offline synthetic relational SQLite databases, tabular datasets
    (CSV/Parquet), and deterministic ground truth evaluation tasks.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate_all_fixtures(self, output_dir: str) -> Dict[str, str]:
        """Generate all databases and tabular files into the target directory."""
        os.makedirs(output_dir, exist_ok=True)
        fixtures = {}

        # SQLite DBs
        fixtures["enterprise_sales_db"] = self.generate_sales_db(
            os.path.join(output_dir, "enterprise_sales.sqlite")
        )
        fixtures["hospital_records_db"] = self.generate_hospital_db(
            os.path.join(output_dir, "hospital_records.sqlite")
        )
        fixtures["financial_ledger_db"] = self.generate_finance_db(
            os.path.join(output_dir, "financial_ledger.sqlite")
        )

        # Tabular datasets
        fixtures["customer_churn_csv"] = self.generate_churn_csv(
            os.path.join(output_dir, "customer_churn.csv")
        )
        fixtures["sensor_telemetry_parquet"] = self.generate_telemetry_parquet(
            os.path.join(output_dir, "sensor_telemetry.parquet")
        )
        fixtures["ecommerce_inventory_csv"] = self.generate_inventory_csv(
            os.path.join(output_dir, "ecommerce_inventory.csv")
        )

        return fixtures

    def generate_sales_db(self, db_path: str) -> str:
        """Generate enterprise sales SQLite database with 6 relational tables."""
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # DDL
        cursor.execute("""
        CREATE TABLE regions (
            region_id INTEGER PRIMARY KEY,
            region_name TEXT NOT NULL,
            country TEXT NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE sales_reps (
            rep_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region_id INTEGER NOT NULL,
            commission_rate REAL NOT NULL,
            hire_date TEXT NOT NULL,
            FOREIGN KEY (region_id) REFERENCES regions(region_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL,
            department TEXT NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            price REAL NOT NULL,
            cost REAL NOT NULL,
            stock_quantity INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            region_id INTEGER NOT NULL,
            tier TEXT NOT NULL,
            credit_limit REAL,
            FOREIGN KEY (region_id) REFERENCES regions(region_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            rep_id INTEGER,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (rep_id) REFERENCES sales_reps(rep_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            discount REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        """)

        # Seed data
        regions = [
            (1, "North America", "USA"),
            (2, "Europe West", "Germany"),
            (3, "Asia Pacific", "Japan"),
            (4, "Latin America", "Brazil")
        ]
        cursor.executemany("INSERT INTO regions VALUES (?, ?, ?);", regions)

        reps = [
            (1, "Alice Walker", 1, 0.08, "2021-03-15"),
            (2, "Bob Martin", 2, 0.07, "2020-07-01"),
            (3, "Chen Wei", 3, 0.09, "2022-01-10"),
            (4, "Daniela Silva", 4, 0.06, "2023-05-20"),
            (5, "Emma Watson", 1, 0.08, "2019-11-01")
        ]
        cursor.executemany("INSERT INTO sales_reps VALUES (?, ?, ?, ?, ?);", reps)

        categories = [
            (1, "Cloud Servers", "Infrastructure"),
            (2, "Developer Tools", "Software"),
            (3, "Enterprise Security", "Security"),
            (4, "Database Licenses", "Software")
        ]
        cursor.executemany("INSERT INTO categories VALUES (?, ?, ?);", categories)

        products = [
            (101, "Compute Cluster Node", 1, 1500.0, 900.0, 45),
            (102, "Storage SAN Array", 1, 3200.0, 2100.0, 18),
            (103, "IDE Team License", 2, 450.0, 150.0, 200),
            (104, "CI/CD Pipeline Enterprise", 2, 1200.0, 400.0, 80),
            (105, "Firewall Appliance X", 3, 2800.0, 1600.0, 25),
            (106, "Endpoint Shield 500", 3, 950.0, 300.0, 110),
            (107, "Distributed SQL Cluster", 4, 4500.0, 2000.0, 15),
            (108, "Graph DB Analytics", 4, 2100.0, 950.0, 30)
        ]
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?);", products)

        customers = [
            (201, "Acme Corp", 1, "Platinum", 50000.0),
            (202, "Beta Systems", 2, "Gold", 25000.0),
            (203, "Gamma Tech", 3, "Silver", 10000.0),
            (204, "Delta Logistics", 4, "Gold", 30000.0),
            (205, "Epsilon AI", 1, "Platinum", 75000.0),
            (206, "Zeta Retail", 2, "Bronze", 5000.0),
            (207, "Eta Robotics", 3, "Gold", 20000.0),
            (208, "Theta Health", 1, "Silver", None) # NULL credit limit
        ]
        cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?);", customers)

        orders = [
            (1001, 201, 1, "2026-01-10", "Shipped", 6200.0),
            (1002, 202, 2, "2026-01-12", "Shipped", 2800.0),
            (1003, 205, 5, "2026-01-15", "Shipped", 12500.0),
            (1004, 203, 3, "2026-01-18", "Pending", 1650.0),
            (1005, 204, 4, "2026-01-20", "Shipped", 4500.0),
            (1006, 201, 1, "2026-01-25", "Shipped", 9000.0),
            (1007, 207, 3, "2026-02-01", "Shipped", 3200.0),
            (1008, 206, 2, "2026-02-05", "Cancelled", 450.0),
            (1009, 208, None, "2026-02-10", "Shipped", 1900.0), # NULL sales rep
            (1010, 205, 5, "2026-02-14", "Shipped", 8000.0)
        ]
        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?);", orders)

        order_items = [
            (5001, 1001, 101, 2, 1500.0, 0.0),
            (5002, 1001, 102, 1, 3200.0, 0.0),
            (5003, 1002, 105, 1, 2800.0, 0.0),
            (5004, 1003, 107, 2, 4500.0, 0.10),
            (5005, 1003, 104, 3, 1200.0, 0.05),
            (5006, 1004, 103, 1, 450.0, 0.0),
            (5007, 1004, 104, 1, 1200.0, 0.0),
            (5008, 1005, 107, 1, 4500.0, 0.0),
            (5009, 1006, 107, 2, 4500.0, 0.0),
            (5010, 1007, 102, 1, 3200.0, 0.0),
            (5011, 1008, 103, 1, 450.0, 0.0),
            (5012, 1009, 106, 2, 950.0, 0.0),
            (5013, 1010, 101, 4, 1500.0, 0.0),
            (5014, 1010, 108, 1, 2000.0, 0.0)
        ]
        cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?);", order_items)

        conn.commit()
        conn.close()
        return db_path

    def generate_hospital_db(self, db_path: str) -> str:
        """Generate hospital records SQLite database with 5 relational tables."""
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute("""
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL,
            floor INTEGER NOT NULL,
            budget REAL NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE doctors (
            doctor_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER NOT NULL,
            specialization TEXT NOT NULL,
            years_experience INTEGER NOT NULL,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            birth_date TEXT NOT NULL,
            gender TEXT NOT NULL,
            blood_type TEXT NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE admissions (
            admission_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            admission_date TEXT NOT NULL,
            discharge_date TEXT,
            diagnosis TEXT NOT NULL,
            room_number INTEGER NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
            FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
        );
        """)
        cursor.execute("""
        CREATE TABLE treatments (
            treatment_id INTEGER PRIMARY KEY,
            admission_id INTEGER NOT NULL,
            treatment_name TEXT NOT NULL,
            cost REAL NOT NULL,
            treatment_date TEXT NOT NULL,
            successful INTEGER NOT NULL,
            FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
        );
        """)

        # Seed data
        depts = [
            (1, "Cardiology", 3, 500000.0),
            (2, "Neurology", 4, 450000.0),
            (3, "Orthopedics", 2, 350000.0),
            (4, "Pediatrics", 1, 300000.0)
        ]
        cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?);", depts)

        doctors = [
            (10, "Dr. Sarah Jenkins", 1, "Interventional Cardiology", 14),
            (11, "Dr. Raj Patel", 1, "Electrophysiology", 8),
            (12, "Dr. Elena Rostova", 2, "Neurosurgery", 18),
            (13, "Dr. Michael Chang", 3, "Joint Replacement", 11),
            (14, "Dr. Olivia Bennett", 4, "Pediatric Care", 6)
        ]
        cursor.executemany("INSERT INTO doctors VALUES (?, ?, ?, ?, ?);", doctors)

        patients = [
            (101, "John Davis", "1965-04-12", "M", "A+"),
            (102, "Mary Wilson", "1982-09-24", "F", "O-"),
            (103, "Robert Johnson", "1958-11-03", "M", "B+"),
            (104, "Linda Martinez", "1974-06-18", "F", "AB+"),
            (105, "William Taylor", "1990-01-30", "M", "O+"),
            (106, "Patricia Anderson", "2015-08-14", "F", "A-")
        ]
        cursor.executemany("INSERT INTO patients VALUES (?, ?, ?, ?, ?);", patients)

        admissions = [
            (1001, 101, 10, "2026-01-05", "2026-01-12", "Coronary Artery Disease", 304),
            (1002, 102, 12, "2026-01-08", "2026-01-15", "Migraine & Neuropathy", 412),
            (1003, 103, 13, "2026-01-10", "2026-01-18", "Osteoarthritis Knee", 205),
            (1004, 104, 11, "2026-01-15", "2026-01-20", "Atrial Fibrillation", 308),
            (1005, 105, 13, "2026-01-22", "2026-01-25", "Meniscus Tear", 210),
            (1006, 106, 14, "2026-02-01", "2026-02-04", "Acute Bronchitis", 102),
            (1007, 101, 10, "2026-02-10", None, "Arrhythmia Followup", 306) # Ongoing admission
        ]
        cursor.executemany("INSERT INTO admissions VALUES (?, ?, ?, ?, ?, ?, ?);", admissions)

        treatments = [
            (501, 1001, "Angioplasty & Stent", 8500.0, "2026-01-06", 1),
            (502, 1001, "Cardiac Rehabilitation", 1200.0, "2026-01-09", 1),
            (503, 1002, "Brain MRI Scan", 2200.0, "2026-01-09", 1),
            (504, 1002, "Nerve Conduction Therapy", 1800.0, "2026-01-11", 1),
            (505, 1003, "Total Knee Arthroplasty", 14500.0, "2026-01-12", 1),
            (506, 1004, "Catheter Ablation", 9800.0, "2026-01-16", 1),
            (507, 1005, "Arthroscopic Knee Repair", 4200.0, "2026-01-23", 1),
            (508, 1006, "Nebulizer Therapy", 650.0, "2026-02-02", 1),
            (509, 1007, "Holter Monitoring", 950.0, "2026-02-11", 1)
        ]
        cursor.executemany("INSERT INTO treatments VALUES (?, ?, ?, ?, ?, ?);", treatments)

        conn.commit()
        conn.close()
        return db_path

    def generate_finance_db(self, db_path: str) -> str:
        """Generate financial double-entry ledger SQLite database with 4 relational tables."""
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.execute("""
        CREATE TABLE currencies (
            code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            symbol TEXT NOT NULL
        );
        """)
        cursor.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            account_number TEXT NOT NULL UNIQUE,
            account_type TEXT NOT NULL,
            currency_code TEXT NOT NULL,
            balance REAL NOT NULL,
            FOREIGN KEY (currency_code) REFERENCES currencies(code)
        );
        """)
        cursor.execute("""
        CREATE TABLE exchange_rates (
            rate_id INTEGER PRIMARY KEY,
            from_currency TEXT NOT NULL,
            to_currency TEXT NOT NULL,
            rate REAL NOT NULL,
            effective_date TEXT NOT NULL,
            FOREIGN KEY (from_currency) REFERENCES currencies(code),
            FOREIGN KEY (to_currency) REFERENCES currencies(code)
        );
        """)
        cursor.execute("""
        CREATE TABLE transactions (
            tx_id INTEGER PRIMARY KEY,
            tx_date TEXT NOT NULL,
            debit_account_id INTEGER NOT NULL,
            credit_account_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            currency_code TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (debit_account_id) REFERENCES accounts(account_id),
            FOREIGN KEY (credit_account_id) REFERENCES accounts(account_id),
            FOREIGN KEY (currency_code) REFERENCES currencies(code)
        );
        """)

        currencies = [
            ("USD", "US Dollar", "$"),
            ("EUR", "Euro", "€"),
            ("GBP", "British Pound", "£"),
            ("JPY", "Japanese Yen", "¥")
        ]
        cursor.executemany("INSERT INTO currencies VALUES (?, ?, ?);", currencies)

        accounts = [
            (1, "1001-CASH-USD", "Asset", "USD", 150000.0),
            (2, "1002-CASH-EUR", "Asset", "EUR", 85000.0),
            (3, "2001-AP-USD", "Liability", "USD", 35000.0),
            (4, "3001-EQUITY", "Equity", "USD", 120000.0),
            (5, "4001-REV-SAAS", "Revenue", "USD", 250000.0),
            (6, "5001-EXP-HOSTING", "Expense", "USD", 45000.0),
            (7, "5002-EXP-SALARY", "Expense", "USD", 115000.0)
        ]
        cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?);", accounts)

        rates = [
            (1, "EUR", "USD", 1.085, "2026-01-01"),
            (2, "GBP", "USD", 1.275, "2026-01-01"),
            (3, "JPY", "USD", 0.0068, "2026-01-01"),
            (4, "USD", "EUR", 0.921, "2026-01-01")
        ]
        cursor.executemany("INSERT INTO exchange_rates VALUES (?, ?, ?, ?, ?);", rates)

        transactions = [
            (101, "2026-01-05", 1, 5, 50000.0, "USD", "Enterprise Annual Subscription", "Settled"),
            (102, "2026-01-10", 6, 1, 12000.0, "USD", "Cloud Infrastructure Hosting", "Settled"),
            (103, "2026-01-15", 7, 1, 45000.0, "USD", "Engineering Payroll Jan", "Settled"),
            (104, "2026-01-20", 2, 5, 25000.0, "EUR", "European SaaS Expansion", "Settled"),
            (105, "2026-01-25", 1, 3, 15000.0, "USD", "Vendor Invoice Settlement", "Settled"),
            (106, "2026-02-01", 1, 5, 35000.0, "USD", "Professional Services Q1", "Settled")
        ]
        cursor.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?);", transactions)

        conn.commit()
        conn.close()
        return db_path

    def generate_churn_csv(self, csv_path: str, n_rows: int = 1000) -> str:
        """Generate customer churn tabular CSV dataset."""
        rng = np.random.default_rng(self.seed)
        customer_ids = [10001 + i for i in range(n_rows)]
        genders = rng.choice(["Male", "Female"], size=n_rows)
        senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.82, 0.18])
        partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
        dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
        tenure_months = rng.integers(1, 73, size=n_rows)
        phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.90, 0.10])
        internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.35, 0.45, 0.20])
        contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
        paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.60, 0.40])
        payment_method = rng.choice([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], size=n_rows)
        
        monthly_charges = np.round(rng.uniform(18.25, 118.75, size=n_rows), 2)
        total_charges = np.round(monthly_charges * tenure_months * rng.uniform(0.95, 1.05, size=n_rows), 2)
        
        # Churn probability based on tenure and contract
        churn_prob = np.where(contract == "Month-to-month", 0.42, 0.11)
        churn_prob = np.where(tenure_months < 12, churn_prob + 0.15, churn_prob)
        churn_prob = np.clip(churn_prob, 0.05, 0.85)
        churn = (rng.random(size=n_rows) < churn_prob).astype(int)

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "gender": genders,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "internet_service": internet_service,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    def generate_telemetry_parquet(self, parquet_path: str, n_rows: int = 10000) -> str:
        """Generate sensor telemetry dataset in Parquet format."""
        rng = np.random.default_rng(self.seed + 1)
        device_ids = [f"SENSOR-{rng.integers(100, 150):03d}" for _ in range(n_rows)]
        
        start_date = datetime(2026, 1, 1, 0, 0, 0)
        timestamps = [start_date + timedelta(seconds=int(i * 10)) for i in range(n_rows)]
        
        temp_c = np.round(rng.normal(65.0, 8.5, size=n_rows), 2)
        pressure_kpa = np.round(rng.normal(101.3, 3.2, size=n_rows), 2)
        vibration_hz = np.round(rng.exponential(4.2, size=n_rows), 2)
        power_w = np.round(rng.uniform(120.0, 450.0, size=n_rows), 1)
        status = rng.choice(["NORMAL", "WARNING", "CRITICAL"], size=n_rows, p=[0.92, 0.06, 0.02])

        df = pd.DataFrame({
            "device_id": device_ids,
            "timestamp": timestamps,
            "temperature_c": temp_c,
            "pressure_kpa": pressure_kpa,
            "vibration_hz": vibration_hz,
            "power_w": power_w,
            "status": status
        })
        df.to_parquet(parquet_path, index=False)
        return parquet_path

    def generate_inventory_csv(self, csv_path: str, n_rows: int = 500) -> str:
        """Generate e-commerce inventory dataset with currency strings and category groupings."""
        rng = np.random.default_rng(self.seed + 2)
        categories = ["Electronics", "Home & Kitchen", "Apparel", "Beauty", "Sports", "Books"]
        
        skus = [f"SKU-{1000 + i}" for i in range(n_rows)]
        chosen_cats = rng.choice(categories, size=n_rows)
        unit_prices = np.round(rng.uniform(5.99, 899.99, size=n_rows), 2)
        price_strs = [f"${p:,.2f}" for p in unit_prices]
        stock_units = rng.integers(0, 500, size=n_rows)
        reorder_points = rng.integers(10, 50, size=n_rows)
        in_stock = (stock_units > 0).astype(int)

        df = pd.DataFrame({
            "sku": skus,
            "category": chosen_cats,
            "price_raw": price_strs,
            "unit_price": unit_prices,
            "stock_quantity": stock_units,
            "reorder_threshold": reorder_points,
            "is_available": in_stock
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    def get_synthetic_benchmark_tasks(self, output_dir: str) -> List[BenchmarkTask]:
        """
        Generate 50+ deterministic benchmark tasks with pre-computed verifiable ground truths
        spanning exact, float_tol, sql_multiset, and dataframe_diff across all benchmarks.
        """
        fixtures = self.generate_all_fixtures(output_dir)
        sales_db = fixtures["enterprise_sales_db"]
        hospital_db = fixtures["hospital_records_db"]
        finance_db = fixtures["financial_ledger_db"]
        churn_csv = fixtures["customer_churn_csv"]
        telemetry_parquet = fixtures["sensor_telemetry_parquet"]
        inventory_csv = fixtures["ecommerce_inventory_csv"]

        tasks: List[BenchmarkTask] = []

        # -------------------------------------------------------------------
        # 1. BIRD-SQL Relational Queries (15 tasks)
        # -------------------------------------------------------------------
        sql_tasks = [
            (
                "syn_sql_001",
                "Find total count of products in category 'Cloud Servers'.",
                sales_db,
                "SELECT COUNT(*) FROM products p JOIN categories c ON p.category_id = c.category_id WHERE c.category_name = 'Cloud Servers';",
                [(2,)],
                {"difficulty": "simple", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_002",
                "List names of sales reps in region 'North America' ordered alphabetically.",
                sales_db,
                "SELECT r.name FROM sales_reps r JOIN regions reg ON r.region_id = reg.region_id WHERE reg.region_name = 'North America' ORDER BY r.name ASC;",
                [("Alice Walker",), ("Emma Watson",)],
                {"difficulty": "simple", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_003",
                "Find the highest product price in the enterprise sales catalog.",
                sales_db,
                "SELECT MAX(price) FROM products;",
                [(4500.0,)],
                {"difficulty": "simple", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_004",
                "Calculate total order amount for customer 'Acme Corp'.",
                sales_db,
                "SELECT SUM(o.total_amount) FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.customer_name = 'Acme Corp';",
                [(15200.0,)],
                {"difficulty": "moderate", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_005",
                "Find all products that have never been ordered.",
                sales_db,
                "SELECT p.product_name FROM products p WHERE p.product_id NOT IN (SELECT DISTINCT product_id FROM order_items);",
                [],
                {"difficulty": "moderate", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_006",
                "Find the customer with the highest number of orders.",
                sales_db,
                "SELECT c.customer_name, COUNT(o.order_id) as cnt FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id ORDER BY cnt DESC, c.customer_name ASC LIMIT 2;",
                [("Acme Corp", 2), ("Epsilon AI", 2)],
                {"difficulty": "moderate", "db_id": "enterprise_sales"}
            ),
            (
                "syn_sql_007",
                "Find all doctors specializing in 'Neurosurgery'.",
                hospital_db,
                "SELECT name, years_experience FROM doctors WHERE specialization = 'Neurosurgery';",
                [("Dr. Elena Rostova", 18)],
                {"difficulty": "simple", "db_id": "hospital_records"}
            ),
            (
                "syn_sql_008",
                "Calculate total cost of all successful treatments in Cardiology department.",
                hospital_db,
                "SELECT SUM(t.cost) FROM treatments t JOIN admissions a ON t.admission_id = a.admission_id JOIN doctors d ON a.doctor_id = d.doctor_id JOIN departments dept ON d.dept_id = dept.dept_id WHERE dept.dept_name = 'Cardiology' AND t.successful = 1;",
                [(20450.0,)],
                {"difficulty": "moderate", "db_id": "hospital_records"}
            ),
            (
                "syn_sql_009",
                "List patient names who have had more than one hospital admission.",
                hospital_db,
                "SELECT p.name FROM patients p JOIN admissions a ON p.patient_id = a.patient_id GROUP BY p.patient_id HAVING COUNT(a.admission_id) > 1;",
                [("John Davis",)],
                {"difficulty": "moderate", "db_id": "hospital_records"}
            ),
            (
                "syn_sql_010",
                "Find the department with the highest total budget.",
                hospital_db,
                "SELECT dept_name, budget FROM departments ORDER BY budget DESC LIMIT 1;",
                [("Cardiology", 500000.0)],
                {"difficulty": "simple", "db_id": "hospital_records"}
            ),
            (
                "syn_sql_011",
                "Find all Asset accounts and their balances in USD.",
                finance_db,
                "SELECT account_number, balance FROM accounts WHERE account_type = 'Asset' AND currency_code = 'USD';",
                [("1001-CASH-USD", 150000.0)],
                {"difficulty": "simple", "db_id": "financial_ledger"}
            ),
            (
                "syn_sql_012",
                "Calculate the sum of all settled USD transactions.",
                finance_db,
                "SELECT SUM(amount) FROM transactions WHERE currency_code = 'USD' AND status = 'Settled';",
                [(157000.0,)],
                {"difficulty": "simple", "db_id": "financial_ledger"}
            ),
            (
                "syn_sql_013",
                "List currency exchange rate from EUR to USD.",
                finance_db,
                "SELECT rate FROM exchange_rates WHERE from_currency = 'EUR' AND to_currency = 'USD';",
                [(1.085,)],
                {"difficulty": "simple", "db_id": "financial_ledger"}
            ),
            (
                "syn_sql_014",
                "Find total expenses recorded across all expense accounts.",
                finance_db,
                "SELECT SUM(balance) FROM accounts WHERE account_type = 'Expense';",
                [(160000.0,)],
                {"difficulty": "simple", "db_id": "financial_ledger"}
            ),
            (
                "syn_sql_015",
                "Find orders with NULL sales representative.",
                sales_db,
                "SELECT order_id, total_amount FROM orders WHERE rep_id IS NULL;",
                [(1009, 1900.0)],
                {"difficulty": "simple", "db_id": "enterprise_sales"}
            )
        ]

        for tid, query, db_f, gsql, gres, meta in sql_tasks:
            meta["golden_sql"] = gsql
            tasks.append(BenchmarkTask(
                task_id=tid,
                benchmark_name="bird_sql",
                query=query,
                db_path=db_f,
                ground_truth=gres,
                eval_type="sql_multiset",
                metadata=meta
            ))

        # -------------------------------------------------------------------
        # 2. InfiAgent-DABench Python Data Analytics Tasks (15 tasks)
        # -------------------------------------------------------------------
        infi_tasks = [
            ("syn_infi_001", "What is the total number of customers in the churn dataset?", churn_csv, 1000, "float_tol", "aggregation"),
            ("syn_infi_002", "What is the mean tenure in months for all customers?", churn_csv, 36.5, "float_tol", "aggregation"),
            ("syn_infi_003", "What is the overall churn rate as a decimal?", churn_csv, 0.28, "float_tol", "statistical_test"),
            ("syn_infi_004", "What is the average monthly charge for customers with Fiber optic internet?", churn_csv, 68.5, "float_tol", "aggregation"),
            ("syn_infi_005", "Are there more female customers than male customers? (True/False)", churn_csv, False, "exact", "boolean"),
            ("syn_infi_006", "What is the maximum total charges value in the dataset?", churn_csv, 8500.0, "float_tol", "aggregation"),
            ("syn_infi_007", "What is the correlation between tenure and monthly charges?", churn_csv, 0.0, "float_tol", "statistical_test"),
            ("syn_infi_008", "What is the most popular payment method?", churn_csv, "Electronic check", "exact", "categorical"),
            ("syn_infi_009", "What is the minimum recorded temperature in the sensor telemetry parquet?", telemetry_parquet, 35.0, "float_tol", "aggregation"),
            ("syn_infi_010", "What is the mean pressure in kPa across all telemetry readings?", telemetry_parquet, 101.3, "float_tol", "aggregation"),
            ("syn_infi_011", "How many total readings are in the sensor telemetry dataset?", telemetry_parquet, 10000, "float_tol", "aggregation"),
            ("syn_infi_012", "What percentage of sensor status readings are NORMAL?", telemetry_parquet, 92.0, "float_tol", "statistical_test"),
            ("syn_infi_013", "What is the total count of distinct sensor device IDs?", telemetry_parquet, 50, "float_tol", "aggregation"),
            ("syn_infi_014", "What is the total inventory valuation (price * stock_quantity)?", inventory_csv, 62500.0, "float_tol", "aggregation"),
            ("syn_infi_015", "How many product SKUs are currently in stock with stock > 0?", inventory_csv, 498, "float_tol", "aggregation"),
        ]

        # Calculate true deterministic values for tabular infi tasks
        churn_df = pd.read_csv(churn_csv)
        telemetry_df = pd.read_parquet(telemetry_parquet)
        inventory_df = pd.read_csv(inventory_csv)

        infi_computed_truths = {
            "syn_infi_001": float(len(churn_df)),
            "syn_infi_002": float(churn_df["tenure_months"].mean()),
            "syn_infi_003": float(churn_df["churn"].mean()),
            "syn_infi_004": float(churn_df[churn_df["internet_service"] == "Fiber optic"]["monthly_charges"].mean()),
            "syn_infi_005": bool((churn_df["gender"] == "Female").sum() > (churn_df["gender"] == "Male").sum()),
            "syn_infi_006": float(churn_df["total_charges"].max()),
            "syn_infi_007": float(churn_df["tenure_months"].corr(churn_df["monthly_charges"])),
            "syn_infi_008": str(churn_df["payment_method"].mode()[0]),
            "syn_infi_009": float(telemetry_df["temperature_c"].min()),
            "syn_infi_010": float(telemetry_df["pressure_kpa"].mean()),
            "syn_infi_011": float(len(telemetry_df)),
            "syn_infi_012": float((telemetry_df["status"] == "NORMAL").mean() * 100.0),
            "syn_infi_013": float(telemetry_df["device_id"].nunique()),
            "syn_infi_014": float((inventory_df["unit_price"] * inventory_df["stock_quantity"]).sum()),
            "syn_infi_015": float((inventory_df["stock_quantity"] > 0).sum())
        }

        for tid, query, tbl, _, etype, qtype in infi_tasks:
            actual_gt = infi_computed_truths[tid]
            tasks.append(BenchmarkTask(
                task_id=tid,
                benchmark_name="infiagent",
                query=query,
                table_path=tbl,
                ground_truth=actual_gt,
                eval_type=etype,
                metadata={"question_type": qtype, "dataset_name": os.path.basename(tbl)}
            ))

        # -------------------------------------------------------------------
        # 3. DataBench Tabular QA Tasks (15 tasks: Scalar, Boolean, List/Set, Table)
        # -------------------------------------------------------------------
        databench_specs = [
            ("syn_db_001", "What is the average tenure of customers on two-year contracts?", churn_csv, "Scalar", "float_tol", float(churn_df[churn_df["contract"] == "Two year"]["tenure_months"].mean())),
            ("syn_db_002", "Is the total number of churned customers greater than 300?", churn_csv, "Boolean", "exact", bool(churn_df["churn"].sum() > 300)),
            ("syn_db_003", "List all distinct contract types in the dataset.", churn_csv, "List/Set", "exact", sorted(list(churn_df["contract"].unique()))),
            ("syn_db_004", "What is the maximum monthly charge in the dataset?", churn_csv, "Scalar", "float_tol", float(churn_df["monthly_charges"].max())),
            ("syn_db_005", "Are there any customers with 0 tenure months?", churn_csv, "Boolean", "exact", bool((churn_df["tenure_months"] == 0).any())),
            ("syn_db_006", "List the distinct internet service categories.", churn_csv, "List/Set", "exact", sorted(list(churn_df["internet_service"].unique()))),
            ("syn_db_007", "What is the standard deviation of vibration_hz in sensor telemetry?", telemetry_parquet, "Scalar", "float_tol", float(telemetry_df["vibration_hz"].std())),
            ("syn_db_008", "Does the telemetry dataset contain any status 'OFFLINE'?", telemetry_parquet, "Boolean", "exact", bool((telemetry_df["status"] == "OFFLINE").any())),
            ("syn_db_009", "List all unique status codes present in sensor telemetry.", telemetry_parquet, "List/Set", "exact", sorted(list(telemetry_df["status"].unique()))),
            ("syn_db_010", "What is the 95th percentile of sensor temperature?", telemetry_parquet, "Scalar", "float_tol", float(np.percentile(telemetry_df["temperature_c"], 95))),
            ("syn_db_011", "Is the average unit price of Beauty products greater than $100?", inventory_csv, "Boolean", "exact", bool(inventory_df[inventory_df["category"] == "Beauty"]["unit_price"].mean() > 100.0)),
            ("syn_db_012", "List all distinct product categories in inventory.", inventory_csv, "List/Set", "exact", sorted(list(inventory_df["category"].unique()))),
            ("syn_db_013", "What is the median stock quantity across all products?", inventory_csv, "Scalar", "float_tol", float(inventory_df["stock_quantity"].median())),
            ("syn_db_014", "Generate a summary table of product count per category.", inventory_csv, "Table", "dataframe_diff", inventory_df.groupby("category").size().reset_index(name="count").to_dict(orient="records")),
            ("syn_db_015", "Generate a breakdown table of mean monthly charges by contract type.", churn_csv, "Table", "dataframe_diff", churn_df.groupby("contract")["monthly_charges"].mean().round(2).reset_index().to_dict(orient="records"))
        ]

        for tid, query, tbl, stype, etype, gt in databench_specs:
            tasks.append(BenchmarkTask(
                task_id=tid,
                benchmark_name="databench",
                query=query,
                table_path=tbl,
                ground_truth=gt,
                eval_type=etype,
                metadata={"semantic_type": stype, "dataset_name": os.path.basename(tbl)}
            ))

        # -------------------------------------------------------------------
        # 4. Pure Hermetic Synthetic Benchmarks (10 tasks)
        # -------------------------------------------------------------------
        synthetic_pure = [
            ("syn_core_001", "Compute (125.5 * 4) + 18.2", None, 520.2, "float_tol"),
            ("syn_core_002", "Is 2026 a leap year?", None, False, "exact"),
            ("syn_core_003", "Normalize string '  NVIDIA NeMo Long-Horizon Eval! ' to canonical title", None, "NVIDIA NeMo Long-Horizon Eval!", "exact"),
            ("syn_core_004", "Calculate sum of squares for [1, 2, 3, 4, 5]", None, 55.0, "float_tol"),
            ("syn_core_005", "What is 15% of 850?", None, 127.5, "float_tol"),
            ("syn_core_006", "Find root of x^2 - 16 = 0 (positive)", None, 4.0, "float_tol"),
            ("syn_core_007", "Evaluate boolean (True and False) or (True and not False)", None, True, "exact"),
            ("syn_core_008", "Sort list ['gamma', 'alpha', 'beta'] alphabetically", None, ["alpha", "beta", "gamma"], "exact"),
            ("syn_core_009", "Calculate 2^10", None, 1024, "float_tol"),
            ("syn_core_010", "What is the float percentage representation of 3/8?", None, 0.375, "float_tol")
        ]

        for tid, query, tbl, gt, etype in synthetic_pure:
            tasks.append(BenchmarkTask(
                task_id=tid,
                benchmark_name="synthetic",
                query=query,
                table_path=tbl,
                ground_truth=gt,
                eval_type=etype,
                metadata={"synthetic_category": "core_reasoning"}
            ))

        return tasks
