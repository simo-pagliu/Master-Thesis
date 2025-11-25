#!/usr/bin/env python3
"""Generate a randomized `alternatives.csv` compatible with the project's loader.

Usage: python randomize_alternatives.py --n 12 --out alternatives_random.csv --seed 42
"""
import csv
import argparse
import random
from pathlib import Path


def read_criteria(file_path):
    criteria = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            name = row[0]
            try:
                minv = int(float(row[2]))
            except Exception:
                minv = 1
            try:
                maxv = int(float(row[3]))
            except Exception:
                maxv = 9
            criteria.append((name, minv, maxv))
    return criteria

import numpy as np
def make_discrete_cell(value):
    # Keep the same string format used in the repo: "{'Discrete': [[<value>]]}"
    return f"{{'Discrete': [[{value[0]}, {value[1]}, {value[2]}]]}}"


def generate(n, criteria, seed=None):
    if seed is not None:
        random.seed(seed)
    rows = []
    for i in range(1, n + 1):
        row = {"name": f"Alternative {i}"}
        for name, minv, maxv in criteria:
            v = [random.randint(minv, maxv), random.randint(minv, maxv), random.randint(minv, maxv)]
            row[name] = make_discrete_cell(v)
        rows.append(row)
    return rows


def write_csv(path, criteria, rows):
    fieldnames = ["name"] + [c[0] for c in criteria]
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="Number of alternatives to generate")
    p.add_argument("--criteria", default="criteria.csv", help="Path to criteria CSV (in Complete Test)")
    p.add_argument("--out", default="alternatives_random.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()

    criteria = read_criteria(args.criteria)
    if not criteria:
        raise SystemExit("No criteria found in criteria file")

    rows = generate(args.n, criteria, seed=args.seed)
    write_csv(args.out, criteria, rows)
    print(f"Wrote {len(rows)} alternatives to {args.out}")


if __name__ == "__main__":
    main()
