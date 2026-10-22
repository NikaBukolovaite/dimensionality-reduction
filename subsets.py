# 3 uždavinys: sudaromos trys imtys iš visos LD1 aibės
# (po 1000 dūžių klasei, be trūkstamų reikšmių – 4837 dūžiai):
#   A – pagrindinė aibė: visi dūžiai, 5 požymiai, išskirtys paliekamos
#       ir tik pažymimos (stulpelis ekstremali_isskirtis), kad jas būtų
#       galima išskirti projekcijose;
#   B – ta pati aibė be RR_sant_post (objektų skaičius nesikeičia);
#   C – A aibė be ekstremalių išskirčių (C yra A poaibis).
#
# Įvestis: data/tiriamoji_pazymeta.csv (sukuria outliers.py)
# Duomenys čia nenormuojami – tai 4 uždavinys.

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "tiriamoji_pazymeta.csv"

SAMPLE_A_FILE = DATA_DIR / "A.csv"
SAMPLE_B_FILE = DATA_DIR / "B.csv"
SAMPLE_C_FILE = DATA_DIR / "C.csv"

CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]

FEATURES_A = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]
REMOVED_FEATURE_B = "RR_sant_post"
FEATURES_B = [feature for feature in FEATURES_A if feature != REMOVED_FEATURE_B]

# Koreliacijos riba, ta pati kaip LD1 (|rho| >= 0,7 – stiprus ryšys).
STRONG_CORRELATION = 0.7

OUTLIER_COLUMN = "ekstremali_isskirtis"

# Identifikatoriai ir žymės paliekami failuose, bet algoritmams nepaduodami.
# "id" vienodas visose imtyse, todėl C objektus galima susieti su A.
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", OUTLIER_COLUMN]


def order_by_class(data):
    # Klasių tvarka N, L, R, V, A, kad lentelės ir grafikai būtų vienodi.
    ordered = data.copy()
    ordered[CLASS_COLUMN] = pd.Categorical(ordered[CLASS_COLUMN], CLASSES, ordered=True)
    ordered = ordered.sort_values(CLASS_COLUMN, kind="stable").reset_index(drop=True)
    ordered[CLASS_COLUMN] = ordered[CLASS_COLUMN].astype(str)
    ordered.insert(0, "id", range(len(ordered)))
    return ordered


def print_correlations(subset_a):
    # B imties pagrindimas: koreliacija skaičiuojama TOJE aibėje, kurią projektuojame.
    correlations = subset_a[FEATURES_A].corr(method="spearman")
    print("Spearman koreliacija tarp požymių A aibėje:")
    print(correlations.round(3).to_string())

    strong_pairs = [
        (first, second, correlations.loc[first, second])
        for i, first in enumerate(FEATURES_A)
        for second in FEATURES_A[i + 1:]
        if abs(correlations.loc[first, second]) >= STRONG_CORRELATION
    ]
    print(f"\nStipriai koreliuotos poros (|rho| >= {STRONG_CORRELATION}):")
    for first, second, value in strong_pairs:
        print(f"  {first} – {second}: rho = {value:.3f}")
    if not strong_pairs:
        print("  nėra")


def build_subsets(data):
    subset_a = data[ID_COLUMNS + FEATURES_A].copy()
    subset_b = data[ID_COLUMNS + FEATURES_B].copy()
    subset_c = subset_a[~subset_a[OUTLIER_COLUMN]].reset_index(drop=True)
    return subset_a, subset_b, subset_c


def print_summary(subset_a, subset_b, subset_c):
    print()
    for name, subset in [("A", subset_a), ("B", subset_b), ("C", subset_c)]:
        features = [column for column in subset.columns if column not in ID_COLUMNS]
        print(f"{name}: {len(subset)} objektų, {len(features)} požymiai: {', '.join(features)}")

    print("\nObjektų skaičius pagal klasę:")
    table = pd.DataFrame({
        "A": subset_a[CLASS_COLUMN].value_counts(),
        "B": subset_b[CLASS_COLUMN].value_counts(),
        "C": subset_c[CLASS_COLUMN].value_counts(),
        "A išskirčių": subset_a.groupby(CLASS_COLUMN)[OUTLIER_COLUMN].sum(),
    }).reindex(CLASSES)
    table.loc["iš viso"] = table.sum()
    print(table.astype(int).to_string())


def check_subsets(subset_a, subset_b, subset_c):
    assert subset_a["id"].equals(subset_b["id"]), "A ir B objektai turi sutapti"
    assert set(subset_c["id"]) <= set(subset_a["id"]), "C turi būti A poaibis"
    assert not subset_c[OUTLIER_COLUMN].any(), "C neturi būti išskirčių"
    assert subset_a[FEATURES_A].notna().all().all(), "A neturi būti trūkstamų reikšmių"
    print("\nPatikros: A ir B objektai sutampa, C yra A poaibis, trūkstamų reikšmių nėra.")


def main():
    data = order_by_class(pd.read_csv(INPUT_FILE))
    subset_a, subset_b, subset_c = build_subsets(data)

    print("3 uždavinys: imtys A, B, C\n")
    print_correlations(subset_a)
    print_summary(subset_a, subset_b, subset_c)
    check_subsets(subset_a, subset_b, subset_c)

    subset_a.to_csv(SAMPLE_A_FILE, index=False)
    subset_b.to_csv(SAMPLE_B_FILE, index=False)
    subset_c.to_csv(SAMPLE_C_FILE, index=False)

    print(f"\nIšsaugota: {SAMPLE_A_FILE}, {SAMPLE_B_FILE}, {SAMPLE_C_FILE}")


if __name__ == "__main__":
    main()