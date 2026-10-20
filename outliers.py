# 2 uždavinys: kiekvienoje klasėje 3×IQR taisykle pažymimos
# ekstremalios išskirtys. Objektas laikomas išskirtimi, jei bent
# vienas jo požymis yra už ribų. Objektai čia nešalinami –
# tik pridedamas stulpelis, pagal kurį vėliau sudaroma C imtis.
#
# Įvestis: data/tiriamoji_pilna.csv (sukuria samples.py)

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "tiriamoji_pilna.csv"
OUTPUT_FILE = DATA_DIR / "tiriamoji_pazymeta.csv"
BOUNDS_FILE = DATA_DIR / "isskirciu_ribos.csv"

CLASS_COLUMN = "klase"
RECORD_COLUMN = "irasas"
RHYTHM_COLUMN = "ritmas"
CLASSES = ["N", "L", "R", "V", "A"]

FEATURES = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]

IQR_MULTIPLIER = 3
OUTLIER_COLUMN = "ekstremali_isskirtis"


def iqr_bounds_table(data):
    # Ribos skaičiuojamos atskirai kiekvienai klasei ir kiekvienam požymiui.
    rows = []
    for class_name in CLASSES:
        class_values = data.loc[data[CLASS_COLUMN] == class_name, FEATURES]
        for feature in FEATURES:
            q1 = class_values[feature].quantile(0.25)
            q3 = class_values[feature].quantile(0.75)
            iqr = q3 - q1
            rows.append({
                CLASS_COLUMN: class_name,
                "pozymis": feature,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "apatine_riba": q1 - IQR_MULTIPLIER * iqr,
                "virsutine_riba": q3 + IQR_MULTIPLIER * iqr,
            })
    return pd.DataFrame(rows)


def outlier_flags(data, bounds):
    # Kiekvienam dūžiui ir požymiui True, jei reikšmė yra už ribų.
    flags = pd.DataFrame(False, index=data.index, columns=FEATURES)
    for row in bounds.itertuples():
        in_class = data[CLASS_COLUMN] == getattr(row, CLASS_COLUMN)
        values = data.loc[in_class, row.pozymis]
        outside = (values < row.apatine_riba) | (values > row.virsutine_riba)
        flags.loc[outside.index, row.pozymis] = outside
    return flags


def print_counts_by_feature(data, flags):
    print("Ekstremalių reikšmių skaičius pagal požymį ir klasę:")
    table = flags.groupby(data[CLASS_COLUMN]).sum().reindex(CLASSES).T
    table["iš_viso"] = table.sum(axis=1)
    print(table.to_string())


def print_counts_by_class(data, is_outlier):
    print("\nObjektai, kuriuose bent vienas požymis už ribų:")
    table = pd.DataFrame({
        "dūžių": data[CLASS_COLUMN].value_counts(),
        "išskirčių": is_outlier.groupby(data[CLASS_COLUMN]).sum(),
    }).reindex(CLASSES)
    table["išskirčių_%"] = (table["išskirčių"] / table["dūžių"] * 100).round(2)
    table.loc["iš viso"] = [len(data), is_outlier.sum(), round(is_outlier.mean() * 100, 2)]
    print(table.to_string())


def print_features_per_outlier(flags, is_outlier):
    print("\nKiek požymių už ribų viename išskirties objekte:")
    print(flags[is_outlier].sum(axis=1).value_counts().sort_index().to_string())


def print_outlier_sources(data, is_outlier):
    # Ar išskirtys susijusios su konkrečiu ritmu ar EKG įrašu.
    outliers = data[is_outlier]

    print("\nIšskirtys pagal ritmą (3 dažniausi kiekvienoje klasėje):")
    for class_name in CLASSES:
        in_class = outliers[outliers[CLASS_COLUMN] == class_name]
        top = in_class[RHYTHM_COLUMN].value_counts().head(3).to_dict()
        print(f"  {class_name}: {top}")

    print("\nIšskirtys pagal EKG įrašą (10 dažniausių):")
    print(outliers.groupby([CLASS_COLUMN, RECORD_COLUMN]).size()
          .sort_values(ascending=False).head(10).to_string())


def main():
    data = pd.read_csv(INPUT_FILE)

    bounds = iqr_bounds_table(data)
    flags = outlier_flags(data, bounds)
    is_outlier = flags.any(axis=1)

    print(f"2 uždavinys: ekstremalios išskirtys ({IQR_MULTIPLIER}×IQR, per klasę), "
          f"n = {len(data)}\n")
    print_counts_by_feature(data, flags)
    print_counts_by_class(data, is_outlier)
    print_features_per_outlier(flags, is_outlier)
    print_outlier_sources(data, is_outlier)

    marked = data.copy()
    marked[OUTLIER_COLUMN] = is_outlier
    marked.to_csv(OUTPUT_FILE, index=False)
    bounds.round(4).to_csv(BOUNDS_FILE, index=False, encoding="utf-8-sig")

    print(f"\nIšsaugota: {OUTPUT_FILE} (stulpelis '{OUTLIER_COLUMN}')")
    print(f"Išsaugota: {BOUNDS_FILE}")


if __name__ == "__main__":
    main()