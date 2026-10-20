# 3 uždavinys: su fiksuota sėkla atrenkama po 100 dūžių kiekvienai
# klasei (iš viso 500) ir sudaromos trys imtys:
#   A – pagrindinė imtis su 5 požymiais;
#   B – ta pati imtis be RR_sant_post (objektų skaičius nesikeičia);
#   C – A imtis be ekstremalių išskirčių (C yra A poaibis).
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
BEATS_PER_CLASS = 100
RANDOM_STATE = 42

FEATURES_A = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]
REMOVED_FEATURE_B = "RR_sant_post"
FEATURES_B = [feature for feature in FEATURES_A if feature != REMOVED_FEATURE_B]

OUTLIER_COLUMN = "ekstremali_isskirtis"

# Identifikatoriai ir žymės paliekami failuose, bet algoritmams nepaduodami.
# "id" vienodas visose imtyse, todėl C objektus galima susieti su A.
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", OUTLIER_COLUMN]


def draw_sample(data):
    sample = (
        data
        .groupby(CLASS_COLUMN, group_keys=False)
        .sample(n=BEATS_PER_CLASS, random_state=RANDOM_STATE)
    )
    # Klasių tvarka N, L, R, V, A, kad lentelės ir grafikai būtų vienodi.
    sample[CLASS_COLUMN] = pd.Categorical(sample[CLASS_COLUMN], CLASSES, ordered=True)
    sample = sample.sort_values(CLASS_COLUMN, kind="stable").reset_index(drop=True)
    sample[CLASS_COLUMN] = sample[CLASS_COLUMN].astype(str)
    sample.insert(0, "id", range(len(sample)))
    return sample


def build_subsets(sample):
    subset_a = sample[ID_COLUMNS + FEATURES_A].copy()
    subset_b = sample[ID_COLUMNS + FEATURES_B].copy()
    subset_c = subset_a[~subset_a[OUTLIER_COLUMN]].reset_index(drop=True)
    return subset_a, subset_b, subset_c


def print_summary(subset_a, subset_b, subset_c):
    print(f"3 uždavinys: imtys (po {BEATS_PER_CLASS} dūžių klasei, "
          f"random_state = {RANDOM_STATE})\n")

    for name, subset in [("A", subset_a), ("B", subset_b), ("C", subset_c)]:
        features = [column for column in subset.columns if column not in ID_COLUMNS]
        print(f"{name}: {len(subset)} objektų, {len(features)} požymiai: {', '.join(features)}")

    print("\nObjektų skaičius pagal klasę:")
    table = pd.DataFrame({
        "A": subset_a[CLASS_COLUMN].value_counts(),
        "B": subset_b[CLASS_COLUMN].value_counts(),
        "C": subset_c[CLASS_COLUMN].value_counts(),
    }).reindex(CLASSES)
    table.loc["iš viso"] = table.sum()
    print(table.to_string())


def check_subsets(subset_a, subset_b, subset_c):
    # Patikros, kad imtys sudarytos teisingai.
    assert subset_a["id"].equals(subset_b["id"]), "A ir B objektai turi sutapti"
    assert set(subset_c["id"]) <= set(subset_a["id"]), "C turi būti A poaibis"
    assert not subset_c[OUTLIER_COLUMN].any(), "C neturi būti išskirčių"
    assert subset_a[FEATURES_A].notna().all().all(), "A neturi būti trūkstamų reikšmių"
    print("\nPatikros: A ir B objektai sutampa, C yra A poaibis, trūkstamų reikšmių nėra.")


def main():
    data = pd.read_csv(INPUT_FILE)

    sample = draw_sample(data)
    subset_a, subset_b, subset_c = build_subsets(sample)

    print_summary(subset_a, subset_b, subset_c)
    check_subsets(subset_a, subset_b, subset_c)

    subset_a.to_csv(SAMPLE_A_FILE, index=False)
    subset_b.to_csv(SAMPLE_B_FILE, index=False)
    subset_c.to_csv(SAMPLE_C_FILE, index=False)

    print(f"\nIšsaugota: {SAMPLE_A_FILE}, {SAMPLE_B_FILE}, {SAMPLE_C_FILE}")


if __name__ == "__main__":
    main()
