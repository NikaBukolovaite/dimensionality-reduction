from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("rezultatai") / "aprasomoji"
SUBSETS = ["A", "C"]

CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]
FEATURES = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]

# Požymių matavimo vienetai lentelėms.
UNITS = {
    "RR_sant_vid": "santykis",
    "RR_pre": "ms",
    "RR_sant_post": "santykis",
    "QRS": "ms",
    "QTc": "ms",
}
STRONG_CORRELATION = 0.7


def overall_statistics(data):
    values = data[FEATURES]
    return pd.DataFrame({
        "Vienetai": pd.Series(UNITS),
        "Min": values.min(),
        "Q1": values.quantile(0.25),
        "Mediana": values.median(),
        "Q3": values.quantile(0.75),
        "Max": values.max(),
        "IQR": values.quantile(0.75) - values.quantile(0.25),
    }).loc[FEATURES]


def medians_by_class(data):
    table = data.groupby(CLASS_COLUMN)[FEATURES].median().reindex(CLASSES).T
    table["Visa imtis"] = data[FEATURES].median()
    return table


def correlation_matrix(data):
    return data[FEATURES].corr(method="spearman")


def show_and_save(table, title, file_name, decimals=3):
    print(f"\n{title}:")
    print(table.round(decimals).to_string())
    table.round(decimals).to_csv(OUTPUT_DIR / file_name, encoding="utf-8-sig")


def print_strong_pairs(correlations):
    print(f"Stipriai koreliuotos poros (|rho| >= {STRONG_CORRELATION}):")
    found = False
    for i, first in enumerate(FEATURES):
        for second in FEATURES[i + 1:]:
            value = correlations.loc[first, second]
            if abs(value) >= STRONG_CORRELATION:
                print(f"  {first} – {second}: rho = {value:.3f}")
                found = True
    if not found:
        print("  nėra")


def compare_a_c(statistics):
    # Kaip pasikeitė statistika pašalinus ekstremalias išskirtis.
    columns = ["Min", "Mediana", "Max", "IQR"]
    table = pd.concat({name: statistics[name][columns] for name in SUBSETS}, axis=1)
    return table


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subsets = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in SUBSETS}
    statistics = {}

    for name, data in subsets.items():
        print(f"\n===== {name} imtis: {len(data)} objektų =====")

        statistics[name] = overall_statistics(data)
        show_and_save(statistics[name], f"1. {name} imties požymių statistika (prieš normavimą)",
                      f"{name}_statistika.csv")
        show_and_save(medians_by_class(data), f"2. {name} imties medianos pagal klasę",
                      f"{name}_medianos_pagal_klase.csv")

        correlations = correlation_matrix(data)
        show_and_save(correlations, f"3. {name} imties Spearman koreliacijos",
                      f"{name}_koreliacijos.csv")
        print_strong_pairs(correlations)

    show_and_save(compare_a_c(statistics),
                  "\n===== A ir C palyginimas (kas pasikeitė pašalinus išskirtis) =====",
                  "A_C_palyginimas.csv")

    print("\nB imtis: tie patys objektai kaip A, be RR_sant_post – statistika sutampa su A.")
    print(f"Išsaugota: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()