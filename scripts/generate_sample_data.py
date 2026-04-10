"""
generate_sample_data.py — Generate a synthetic Titanic-like dataset for daily batch scoring.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --rows 200 --output data/raw/daily_input.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate(n_rows: int = 100, seed: int = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pclass   = rng.choice([1, 2, 3], size=n_rows, p=[0.24, 0.21, 0.55])
    sex      = rng.choice(["male", "female"], size=n_rows, p=[0.65, 0.35])
    age      = rng.choice(
        np.concatenate([rng.uniform(1, 15, 15), rng.uniform(16, 60, 75), rng.uniform(61, 80, 10)]),
        size=n_rows, replace=True,
    ).round(1)
    # ~5% missing age
    age = np.where(rng.random(n_rows) < 0.05, np.nan, age)

    sibsp    = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.68, 0.23, 0.06, 0.02, 0.01])
    parch    = rng.choice([0, 1, 2, 3],    size=n_rows, p=[0.76, 0.13, 0.09, 0.02])

    # Fare correlated with Pclass
    fare_base = {1: 80.0, 2: 20.0, 3: 10.0}
    fare = np.array([
        max(0, rng.normal(fare_base[p], fare_base[p] * 0.4))
        for p in pclass
    ]).round(2)

    # Cabin: ~77% missing, rest get a deck letter
    decks = ["A", "B", "C", "D", "E", "F", "G"]
    cabin_values = [f"{rng.choice(decks)}{rng.integers(1, 99)}" for _ in range(n_rows)]
    cabin = [None if rng.random() < 0.77 else v for v in cabin_values]

    embarked = rng.choice(["S", "C", "Q"], size=n_rows, p=[0.72, 0.19, 0.09]).tolist()
    embarked = [None if rng.random() < 0.002 else v for v in embarked]

    # Realistic name titles
    def make_name(i, s):
        first  = ["James", "John", "Mary", "Alice", "Robert", "Henry", "Anna", "Thomas"][i % 8]
        last   = ["Smith", "Brown", "Jones", "Taylor", "Wilson", "Davis", "White", "Hall"][i % 8]
        if s == "female":
            title = rng.choice(["Mrs", "Miss", "Ms", "Mme"], p=[0.45, 0.45, 0.05, 0.05])
        else:
            title = rng.choice(["Mr", "Dr", "Rev", "Master"], p=[0.88, 0.06, 0.03, 0.03])
        return f"{last}, {title}. {first}"

    names   = [make_name(i, sex[i]) for i in range(n_rows)]
    tickets = [f"SYN{rng.integers(10000, 99999)}" for _ in range(n_rows)]

    df = pd.DataFrame({
        "PassengerId": range(9000, 9000 + n_rows),
        "Pclass":      pclass,
        "Name":        names,
        "Sex":         sex,
        "Age":         age,
        "SibSp":       sibsp,
        "Parch":       parch,
        "Ticket":      tickets,
        "Fare":        fare,
        "Cabin":       cabin,
        "Embarked":    embarked,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Titanic-like data")
    parser.add_argument("--rows",   type=int, default=100,                      help="Number of rows to generate")
    parser.add_argument("--output", default="data/raw/daily_input.csv",         help="Output CSV path")
    parser.add_argument("--seed",   type=int, default=None,                     help="Random seed (omit for different data each run)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_rows=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} rows → {args.output}")
    print(f"  Age missing:      {df['Age'].isnull().sum()}")
    print(f"  Cabin missing:    {df['Cabin'].isnull().sum()}")
    print(f"  Embarked missing: {df['Embarked'].isnull().sum()}")
