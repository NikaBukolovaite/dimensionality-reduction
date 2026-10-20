# 4 uždavinys: kiekviena imtis normuojama atspariuoju mastelio keitimu
# (robust scaling): x' = (x - mediana) / IQR.
#
# Kiekviena imtis normuojama pagal SAVO medianą ir IQR, nes kiekviena
# imtis yra atskiras eksperimentas (pvz., C imtyje be išskirčių IQR mažesnis).
# Normuojami tik požymiai. Identifikatoriai (id, irasas, R), klasė,
# ritmas ir išskirties žymė lieka nepakeisti – algoritmams jie nepaduodami.
# T_tipas į imtis neįtrauktas (3 uždavinys), todėl čia jo nėra.
#
# Įvestis: data/A.csv, data/B.csv, data/C.csv (sukuria subsets.py)

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
SUBSETS = ["A", "B", "C"]
PARAMETERS_FILE = DATA_DIR / "normavimo_parametrai.csv"

CLASS_COLUMN = "klase"
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", "ekstremali_isskirtis"]


def feature_columns(subset):
    return [column for column in subset.columns if column not in ID_COLUMNS]


def robust_scaling(values):
    median = values.median()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    return (values - median) / iqr, median, iqr


def normalize_subset(subset, subset_name):
    normalized = subset.copy()
    parameters = []

    for feature in feature_columns(subset):
        normalized[feature], median, iqr = robust_scaling(subset[feature])
        parameters.append({
            "imtis": subset_name,
            "pozymis": feature,
            "mediana": median,
            "IQR": iqr,
        })

    return normalized, pd.DataFrame(parameters)


def check_normalized(normalized):
    # Po normavimo kiekvieno požymio mediana = 0, IQR = 1.
    features = normalized[feature_columns(normalized)]
    medians = features.median()
    iqrs = features.quantile(0.75) - features.quantile(0.25)
    assert (medians.abs() < 1e-9).all(), "Mediana po normavimo turi būti 0"
    assert ((iqrs - 1).abs() < 1e-9).all(), "IQR po normavimo turi būti 1"


def print_ranges(normalized, subset_name):
    features = normalized[feature_columns(normalized)]
    table = features.agg(["min", "median", "max"]).T.round(2)
    print(f"\n{subset_name} imtis po normavimo (min, mediana, max):")
    print(table.to_string())


def main():
    print("4 uždavinys: atsparusis mastelio keitimas (x - mediana) / IQR")

    all_parameters = []

    for subset_name in SUBSETS:
        subset = pd.read_csv(DATA_DIR / f"{subset_name}.csv")
        normalized, parameters = normalize_subset(subset, subset_name)

        check_normalized(normalized)
        print_ranges(normalized, subset_name)

        normalized.to_csv(DATA_DIR / f"{subset_name}_norm.csv", index=False)
        all_parameters.append(parameters)

    parameters = pd.concat(all_parameters, ignore_index=True)
    print("\nNormavimo parametrai (mediana ir IQR pagal imtį):")
    print(parameters.pivot(index="pozymis", columns="imtis",
                           values=["mediana", "IQR"]).round(3).to_string())
    parameters.round(4).to_csv(PARAMETERS_FILE, index=False, encoding="utf-8-sig")

    print("\nPatikros: kiekvienoje imtyje požymių mediana = 0, IQR = 1.")
    print(f"Išsaugota: " + ", ".join(f"data/{name}_norm.csv" for name in SUBSETS))
    print(f"Išsaugota: {PARAMETERS_FILE}")


if __name__ == "__main__":
    main()
