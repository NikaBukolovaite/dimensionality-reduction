# 16–17 uždaviniai (t-SNE dalis): projekcijos jautrumas duomenų paruošimui.
#
# Visos trys imtys projektuojamos su A aibėje parinktais parametrais
# (perplexity = 50, PCA inicializacija). Su PCA inicializacija t-SNE
# rezultatas nuo sėklos nepriklauso (patikrinta tsne_stability.py), todėl
# kiekvienai imčiai pakanka vieno paleidimo.
#
# Atskaitos taškas – kiek skiriasi dvi A projekcijos su atsitiktine
# inicializacija ir skirtingomis sėklomis (natūrali t-SNE sklaida).
#
# A ir B (objektai tie patys, B be RR_sant_post):
#   T, C, NH, rho pokytis; rho(D_A, D_B); k kaimynų sutapimas.
# A ir C (C be ekstremalių išskirčių):
#   1) A projekcija su visais objektais; 2) iš jos paliekami tik bendri
#   objektai (pagal id); 3) C projekcija skaičiuojama iš naujo;
#   4) lyginami bendrų objektų atstumai ir kaimynystės.
#
# Įvestis: data/A_norm.csv, data/B_norm.csv, data/C_norm.csv,
#          rezultatai/tsne/stabilumas/*.npy (iš tsne_stability.py)

import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from itertools import combinations
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TSNE"))

from metrics import evaluate_projection, neighbor_overlap
from tsne_params import load_data, run_tsne

TSNE_DIR = ROOT / "rezultatai" / "tsne"
STABILITY_DIR = TSNE_DIR / "stabilumas"
COMPARE_DIR = TSNE_DIR / "jautrumas"

CLASS_COLUMN = "klase"
CHOSEN = {"perplexity": 50, "init": "pca"}
CHOSEN_NAME = f"p{CHOSEN['perplexity']}_{CHOSEN['init']}"
RANDOM_STATE = 0
BASELINE_SEEDS = [0, 1, 2, 3, 4]
OVERLAP_K = [5, 15, 50]
SHOWN_METRICS = ["T_15", "C_15", "NH_10", "Spearman_rho", "Stress_1", "kNN_tikslumas"]


def project(X, subset_name):
    path = COMPARE_DIR / f"tsne_{subset_name}_{CHOSEN_NAME}.npy"
    if path.exists():
        return np.load(path)
    Y = run_tsne(X, CHOSEN["perplexity"], CHOSEN["init"], RANDOM_STATE)
    np.save(path, Y)
    return Y


def distance_rho(first, second):
    rho, _ = spearmanr(pdist(first), pdist(second))
    return rho


def overlaps(first, second):
    return {f"kaimynai_{k}": neighbor_overlap(first, second, k) for k in OVERLAP_K}


def seed_baseline(common_rows=None):
    # Atsitiktinės inicializacijos A projekcijos su skirtingomis sėklomis.
    projections = {seed: np.load(STABILITY_DIR / f"tsne_p{CHOSEN['perplexity']}_random_seed{seed}.npy")
                   for seed in BASELINE_SEEDS}
    rows = []
    for first, second in combinations(BASELINE_SEEDS, 2):
        Y1, Y2 = projections[first], projections[second]
        if common_rows is not None:
            Y1, Y2 = Y1[common_rows], Y2[common_rows]
        rows.append({"rho_projekciju": distance_rho(Y1, Y2), **overlaps(Y1, Y2)})
    return pd.DataFrame(rows).mean()


def quality_table(first_name, first, second_name, second):
    table = pd.DataFrame({first_name: first, second_name: second}).loc[SHOWN_METRICS]
    table["pokytis"] = table[second_name] - table[first_name]
    return table


def similarity_table(similarity, baseline, name):
    columns = ["rho_projekciju"] + [f"kaimynai_{k}" for k in OVERLAP_K]
    return pd.DataFrame({
        name: [similarity[column] for column in columns],
        "atsitiktinė init.: A sėkla prieš A sėklą": [baseline[column] for column in columns],
    }, index=columns)


def save_and_print(table, title, file_name):
    print(f"\n{title}:")
    print(table.round(3).to_string())
    table.round(4).to_csv(COMPARE_DIR / file_name, encoding="utf-8-sig")


def main():
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print("16–17 uždaviniai: t-SNE projekcijos jautrumas duomenų paruošimui")
    print(f"Parametrai: {CHOSEN}")

    data_a, X_a, labels_a, _ = load_data(ROOT / "data" / "A_norm.csv")
    data_b, X_b, labels_b, _ = load_data(ROOT / "data" / "B_norm.csv")
    data_c, X_c, labels_c, _ = load_data(ROOT / "data" / "C_norm.csv")

    start = time.time()
    Y_a = project(X_a, "A")
    Y_b = project(X_b, "B")
    Y_c = project(X_c, "C")
    print(f"Projekcijos apskaičiuotos ({time.time() - start:.0f} s)")

    # --- A ir B ---------------------------------------------------------
    print("\n=== A ir B (be RR_sant_post), objektų skaičius nesikeičia ===")
    rho_space = distance_rho(X_a, X_b)
    space_overlap = overlaps(X_a, X_b)
    print(f"Pradinės erdvės: rho(D_A, D_B) = {rho_space:.3f}; "
          + ", ".join(f"{name} = {value:.3f}" for name, value in space_overlap.items()))

    metrics_a = evaluate_projection(X_a, Y_a, labels_a)
    metrics_b = evaluate_projection(X_b, Y_b, labels_b)
    save_and_print(quality_table("A", metrics_a, "B", metrics_b),
                   "A ir B projekcijų kokybė", "A_B_kokybe.csv")
    similarity_ab = {"rho_projekciju": distance_rho(Y_a, Y_b), **overlaps(Y_a, Y_b)}
    save_and_print(similarity_table(similarity_ab, seed_baseline(), "A prieš B"),
                   f"A ir B projekcijų panašumas; pradinės erdvės rho(D_A, D_B) = {rho_space:.3f}",
                   "A_B_panasumas.csv")

    # --- A ir C ---------------------------------------------------------
    print("\n=== A ir C (be ekstremalių išskirčių), lyginami tik bendri objektai ===")
    position_in_a = pd.Series(np.arange(len(data_a)), index=data_a["id"])
    common_rows = position_in_a.loc[data_c["id"]].to_numpy()
    print(f"A: {len(data_a)} objektų, C: {len(data_c)}, bendri: {len(common_rows)}")

    Y_a_common = Y_a[common_rows]
    metrics_a_common = evaluate_projection(X_a[common_rows], Y_a_common, labels_c)
    metrics_c = evaluate_projection(X_c, Y_c, labels_c)
    save_and_print(quality_table("A_bendri", metrics_a_common, "C", metrics_c),
                   "A (tik bendri objektai) ir C projekcijų kokybė", "A_C_kokybe.csv")
    similarity_ac = {"rho_projekciju": distance_rho(Y_a_common, Y_c), **overlaps(Y_a_common, Y_c)}
    save_and_print(similarity_table(similarity_ac, seed_baseline(common_rows), "A bendri prieš C"),
                   "A ir C projekcijų panašumas (bendri objektai)", "A_C_panasumas.csv")

    print(f"\nLentelės: {COMPARE_DIR}")


if __name__ == "__main__":
    main()
