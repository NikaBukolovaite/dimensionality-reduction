# 1 uždavinys: iš LD1 sutvarkytos aibės pašalinami dūžiai,
# kuriuose trūksta bent vieno atrinkto požymio reikšmės.

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "tiriamoji_sutvarkyta.csv"
COMPLETE_FILE = DATA_DIR / "tiriamoji_pilna.csv"

CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]

FEATURES = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]


def missing_values_table(data):
    missing = data[FEATURES].isna()
    return pd.DataFrame({
        "trūksta": missing.sum(),
        "dalis_%": (missing.mean() * 100).round(2),
    })


def rows_by_class_table(data, has_missing):
    table = pd.DataFrame({
        "iš_viso": data[CLASS_COLUMN].value_counts(),
        "pašalinta": has_missing.groupby(data[CLASS_COLUMN]).sum(),
    }).reindex(CLASSES)
    table["liko"] = table["iš_viso"] - table["pašalinta"]
    table["pašalinta_%"] = (table["pašalinta"] / table["iš_viso"] * 100).round(2)
    return table


def remove_incomplete_rows(data):
    has_missing = data[FEATURES].isna().any(axis=1)

    print("1 uždavinys: dūžių be trūkstamų reikšmių atranka\n")
    print("Trūkstamos reikšmės atrinktuose požymiuose:")
    print(missing_values_table(data).to_string())

    print(f"\nDūžiai su bent viena trūkstama reikšme: {has_missing.sum()} "
          f"iš {len(data)} ({has_missing.mean() * 100:.2f} %)")

    print("\nPagal klasę:")
    print(rows_by_class_table(data, has_missing).to_string())

    return data[~has_missing].reset_index(drop=True)


def main():
    data = pd.read_csv(INPUT_FILE)

    complete = remove_incomplete_rows(data)
    complete.to_csv(COMPLETE_FILE, index=False)

    print(f"\nIšsaugota: {COMPLETE_FILE} ({len(complete)} dūžiai)")


if __name__ == "__main__":
    main()