from pathlib import Path
import numpy as np
import pandas as pd


# Project paths and configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw"

CUSTOMERS_FILE = RAW_PATH / "customers" / "telco_customers.csv"
BILLING_FILE = RAW_PATH / "billing" / "billing_transactions.json"
COMPLAINTS_FILE = RAW_PATH / "complaints" / "customer_complaints.csv"
NETWORK_FILE = RAW_PATH / "network_events" / "network_events.json"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def create_directories():
    """Create output directories when they do not already exist."""
    BILLING_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMPLAINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    NETWORK_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_customers():
    """Load and validate the CRM customer source."""
    if not CUSTOMERS_FILE.exists():
        raise FileNotFoundError(
            f"Customer file was not found: {CUSTOMERS_FILE}\n"
            "Download telco_customers.csv before running this script."
        )

    customers = pd.read_csv(CUSTOMERS_FILE)

    required_columns = {
        "customerID",
        "MonthlyCharges",
        "PaymentMethod",
        "Churn",
    }
    missing_columns = required_columns.difference(customers.columns)
    if missing_columns:
        raise ValueError(
            f"Customer source is missing columns: {sorted(missing_columns)}"
        )

    customers["MonthlyCharges"] = pd.to_numeric(
        customers["MonthlyCharges"], errors="coerce"
    )
    customers["MonthlyCharges"] = customers["MonthlyCharges"].fillna(
        customers["MonthlyCharges"].median()
    )

    customers["churn_flag"] = (
        customers["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )

    print(f"Loaded {len(customers):,} customer records.")
    return customers


def generate_billing(customers):
    """Generate 12 months of billing transactions in JSON Lines format."""
    billing_months = pd.date_range(
        start="2025-09-01",
        periods=12,
        freq="MS",
    )

    billing = customers[
        ["customerID", "MonthlyCharges", "PaymentMethod", "churn_flag"]
    ].merge(
        pd.DataFrame({"billing_month": billing_months}),
        how="cross",
    )

    record_count = len(billing)

    billing["usage_charges"] = rng.uniform(0, 25, record_count).round(2)
    billing["discount_amount"] = rng.choice(
        [0, 0, 0, 5, 10, 15],
        size=record_count,
    )

    taxable_amount = (
        billing["MonthlyCharges"]
        + billing["usage_charges"]
        - billing["discount_amount"]
    ).clip(lower=0)

    billing["tax_amount"] = (taxable_amount * 0.14).round(2)
    billing["total_amount"] = (
        taxable_amount + billing["tax_amount"]
    ).round(2)

    # Churned customers are deliberately more likely to pay late or not pay.
    is_churned = billing["churn_flag"].eq(1)
    paid_probability = np.where(is_churned, 0.70, 0.90)
    late_probability = np.where(is_churned, 0.20, 0.08)
    random_values = rng.random(record_count)

    billing["payment_status"] = np.select(
        [
            random_values < paid_probability,
            random_values < (paid_probability + late_probability),
        ],
        ["Paid", "Late"],
        default="Unpaid",
    )

    billing["invoice_id"] = [
        f"INV-{number:07d}"
        for number in range(1, record_count + 1)
    ]

    billing["payment_date"] = (
        billing["billing_month"]
        + pd.to_timedelta(
            rng.integers(1, 35, record_count),
            unit="D",
        )
    )

    billing.loc[
        billing["payment_status"].eq("Unpaid"),
        "payment_date",
    ] = pd.NaT

    billing = billing[
        [
            "invoice_id",
            "customerID",
            "billing_month",
            "MonthlyCharges",
            "usage_charges",
            "discount_amount",
            "tax_amount",
            "total_amount",
            "PaymentMethod",
            "payment_status",
            "payment_date",
        ]
    ]

    billing.to_json(
        BILLING_FILE,
        orient="records",
        lines=True,
        date_format="iso",
    )

    print(f"Generated {len(billing):,} billing transactions.")


def generate_complaints(customers):
    """Generate customer complaint records in CSV format."""
    complaint_categories = [
        "Network Quality",
        "Billing",
        "Internet Speed",
        "Service Outage",
        "Customer Service",
    ]
    priorities = ["Low", "Medium", "High", "Critical"]
    statuses = ["Closed", "Resolved", "In Progress", "Escalated"]
    channels = ["Call Center", "Mobile App", "Website", "Branch"]

    complaint_records = []
    complaint_number = 1

    for customer in customers.itertuples(index=False):
        average_complaints = 1.8 if customer.churn_flag == 1 else 0.45
        complaint_count = rng.poisson(average_complaints)

        for _ in range(complaint_count):
            opened_date = pd.Timestamp("2025-09-01") + pd.Timedelta(
                days=int(rng.integers(0, 365)),
                hours=int(rng.integers(0, 24)),
            )

            priority = rng.choice(
                priorities,
                p=[0.20, 0.40, 0.30, 0.10],
            )

            resolution_hours = round(float(rng.uniform(2, 96)), 2)
            satisfaction_score = int(
                rng.integers(1, 4)
                if customer.churn_flag == 1
                else rng.integers(3, 6)
            )

            complaint_records.append(
                {
                    "complaint_id": f"CMP-{complaint_number:06d}",
                    "customerID": customer.customerID,
                    "opened_at": opened_date,
                    "category": rng.choice(complaint_categories),
                    "priority": priority,
                    "status": rng.choice(statuses),
                    "channel": rng.choice(channels),
                    "resolution_hours": resolution_hours,
                    "satisfaction_score": satisfaction_score,
                }
            )
            complaint_number += 1

    complaints = pd.DataFrame(complaint_records)
    complaints.to_csv(COMPLAINTS_FILE, index=False)

    print(f"Generated {len(complaints):,} complaint records.")


def generate_network_events(customers):
    """Generate network-monitoring events in JSON Lines format."""
    event_count = 30_000

    customer_weights = np.where(
        customers["churn_flag"].eq(1),
        2.5,
        1.0,
    )
    customer_weights = customer_weights / customer_weights.sum()

    sampled_customers = rng.choice(
        customers["customerID"],
        size=event_count,
        p=customer_weights,
    )

    event_types = rng.choice(
        [
            "Normal",
            "High Latency",
            "Packet Loss",
            "Service Degradation",
            "Outage",
        ],
        size=event_count,
        p=[0.55, 0.15, 0.12, 0.13, 0.05],
    )

    network_events = pd.DataFrame(
        {
            "event_id": [
                f"NET-{number:08d}"
                for number in range(1, event_count + 1)
            ],
            "customerID": sampled_customers,
            "event_timestamp": (
                pd.Timestamp("2025-09-01")
                + pd.to_timedelta(
                    rng.integers(0, 365 * 24 * 60, event_count),
                    unit="m",
                )
            ),
            "event_type": event_types,
            "region": rng.choice(
                ["Cairo", "Giza", "Alexandria", "Delta", "Upper Egypt"],
                event_count,
            ),
            "tower_id": [
                f"TOWER-{number:04d}"
                for number in rng.integers(1, 501, event_count)
            ],
        }
    )

    network_events["latency_ms"] = np.where(
        network_events["event_type"].eq("Normal"),
        rng.uniform(15, 55, event_count),
        rng.uniform(70, 350, event_count),
    ).round(2)

    network_events["packet_loss_pct"] = np.where(
        network_events["event_type"].isin(
            ["Packet Loss", "Service Degradation", "Outage"]
        ),
        rng.uniform(5, 40, event_count),
        rng.uniform(0, 3, event_count),
    ).round(2)

    network_events["download_speed_mbps"] = np.where(
        network_events["event_type"].eq("Normal"),
        rng.uniform(40, 200, event_count),
        rng.uniform(1, 35, event_count),
    ).round(2)

    network_events["outage_minutes"] = np.where(
        network_events["event_type"].eq("Outage"),
        rng.integers(5, 240, event_count),
        0,
    )

    network_events.to_json(
        NETWORK_FILE,
        orient="records",
        lines=True,
        date_format="iso",
    )

    print(f"Generated {len(network_events):,} network events.")


def main():
    create_directories()
    customers = load_customers()
    generate_billing(customers)
    generate_complaints(customers)
    generate_network_events(customers)
    print("\nAll data sources were generated successfully.")


if __name__ == "__main__":
    main()
