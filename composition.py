# 5 uždavinys: trijų imčių sudėties aprašymas ataskaitai.
#   1 lentelė – objektų skaičius pagal klasę kiekvienoje imtyje;
#   2 lentelė – pašalintos ekstremalios išskirtys pagal klasę (A -> C);
#   3 lentelė – pašalintų išskirčių ritmai;
#   4 lentelė – EKG įrašų skaičius pagal klasę.
#
# Įvestis: data/A.csv, data/B.csv, data/C.csv (sukuria subsets.py)

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "imciu_sudetis"

CLASS_COLUMN = "klase"
RECORD_COLUMN = "irasas"
RHYTHM_COLUMN = "ritmas"
OUTLIER_COLUMN = "ekstremali_isskirtis"
CLASSES = ["N", "L", "R", "V", "A"]
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, RHYTHM_COLUMN, OUTLIER_COLUMN]


def counts_by_class(subsets):
    table = pd.DataFrame({
        name: subset[CLASS_COLUMN].value_counts() for name, subset in subsets.items()
    }).reindex(CLASSES)
    table.loc["Iš viso"] = table.sum()

    features = pd.Series({
        name: len([column for column in subset.columns if column not in ID_COLUMNS])
        for name, subset in subsets.items()
    })
    table.loc["Požymių"] = features
    return table.astype(int)


def removed_outliers(subset_a, subset_c):
    before = subset_a[CLASS_COLUMN].value_counts()
    after = subset_c[CLASS_COLUMN].value_counts()

    table = pd.DataFrame({"A": before, "C": after}).reindex(CLASSES)
    table["Pašalinta"] = table["A"] - table["C"]
    table.loc["Iš viso"] = table.sum()
    table["Pašalinta, %"] = (table["Pašalinta"] / table["A"] * 100).round(1)
    return table


def removed_rhythms(subset_a):
    outliers = subset_a[subset_a[OUTLIER_COLUMN]]
    table = pd.crosstab(outliers[RHYTHM_COLUMN], outliers[CLASS_COLUMN])
    table = table.reindex(columns=CLASSES, fill_value=0)
    table["Iš viso"] = table.sum(axis=1)
    return table.sort_values("Iš viso", ascending=False)


def records_by_class(subsets):
    table = pd.DataFrame({
        name: subset.groupby(CLASS_COLUMN)[RECORD_COLUMN].nunique()
        for name, subset in subsets.items()
    }).reindex(CLASSES)
    return table.astype(int)


def show_and_save(table, title, file_name):
    print(f"\n{title}:")
    print(table.to_string())
    table.to_csv(OUTPUT_DIR / file_name, encoding="utf-8-sig")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subsets = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ["A", "B", "C"]}

    print("5 uždavinys: imčių sudėtis")

    show_and_save(counts_by_class(subsets),
                  "Objektų skaičius pagal klasę", "objektai_pagal_klase.csv")
    show_and_save(removed_outliers(subsets["A"], subsets["C"]),
                  "Pašalintos ekstremalios išskirtys (A -> C)", "pasalintos_isskirtys.csv")
    show_and_save(removed_rhythms(subsets["A"]),
                  "Pašalintų išskirčių ritmai", "isskirciu_ritmai.csv")
    show_and_save(records_by_class(subsets),
                  "EKG įrašų skaičius pagal klasę", "irasai_pagal_klase.csv")

    print(f"\nLentelės išsaugotos: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
