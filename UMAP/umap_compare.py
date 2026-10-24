from itertools import combinations
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import evaluate_projection, neighbor_overlap
from umap_plots import metric_line, parameters_title, plot_projection

warnings.filterwarnings("ignore", message="n_jobs value")

DATA_DIR = ROOT / "data"
UMAP_DIR = ROOT / "rezultatai" / "umap"
STABILITY_DIR = UMAP_DIR / "stabilumas"
COMPARE_DIR = UMAP_DIR / "jautrumas"
FIGURES_DIR = UMAP_DIR / "grafikai" / "jautrumas"

CLASS_COLUMN = "klase"
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", "ekstremali_isskirtis"]

CHOSEN = {"n_neighbors": 50, "min_dist": 0.1, "init": "spectral"}
CHOSEN_NAME = f"nn{CHOSEN['n_neighbors']}_md{CHOSEN['min_dist']}_{CHOSEN['init']}"
SEEDS = [0, 1, 2, 3, 4]
OVERLAP_K = [5, 15, 50]
LOCAL_METRICS = ["T_15", "C_15", "NH_10", "Spearman_rho", "Stress_1", "kNN_tikslumas"]


# Duomenys ir projekcijos
def load_subset(name):
    data = pd.read_csv(DATA_DIR / f"{name}_norm.csv")
    features = [column for column in data.columns if column not in ID_COLUMNS]
    return data, data[features].to_numpy()


def project_a(seed):
    # A projekcijos jau suskaičiuotos umap_stability.py – nekartojame.
    return np.load(STABILITY_DIR / f"umap_{CHOSEN_NAME}_seed{seed}.npy")


def project(X, subset_name, seed):
    path = COMPARE_DIR / f"umap_{subset_name}_{CHOSEN_NAME}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    Y = umap.UMAP(n_components=2, random_state=seed, **CHOSEN).fit_transform(X)
    np.save(path, Y)
    return Y


def distance_rho(first, second):
    rho, _ = spearmanr(pdist(first), pdist(second))
    return rho


def overlaps(first, second):
    return {f"kaimynai_{k}": neighbor_overlap(first, second, k) for k in OVERLAP_K}


# Atskaitos taškas: kiek skiriasi A projekcijos su skirtingomis sėklomis
def seed_baseline(projections_a, common_rows=None):
    rows = []
    for first, second in combinations(SEEDS, 2):
        Y1, Y2 = projections_a[first], projections_a[second]
        if common_rows is not None:
            Y1, Y2 = Y1[common_rows], Y2[common_rows]
        rows.append({"rho_projekciju": distance_rho(Y1, Y2), **overlaps(Y1, Y2)})
    return pd.DataFrame(rows).mean()


# A ir B
def compare_a_b(data_a, X_a, X_b, projections_a):
    print("\n=== A ir B (be RR_sant_post), objektų skaičius nesikeičia ===")
    rho_space = distance_rho(X_a, X_b)
    space_overlap = overlaps(X_a, X_b)
    print(f"Pradinės erdvės: rho(D_A, D_B) = {rho_space:.3f}; "
          + ", ".join(f"{name} = {value:.3f}" for name, value in space_overlap.items()))

    labels = data_a[CLASS_COLUMN].to_numpy()
    rows = []
    for seed in SEEDS:
        start = time.time()
        Y_a = projections_a[seed]
        Y_b = project(X_b, "B", seed)
        metrics_a = evaluate_projection(X_a, Y_a, labels)
        metrics_b = evaluate_projection(X_b, Y_b, labels)
        row = {"seed": seed}
        for metric in LOCAL_METRICS:
            row[f"A_{metric}"] = metrics_a[metric]
            row[f"B_{metric}"] = metrics_b[metric]
        row["rho_projekciju"] = distance_rho(Y_a, Y_b)
        row.update(overlaps(Y_a, Y_b))
        rows.append(row)
        print(f"  sėkla {seed}: B T_15={metrics_b['T_15']:.3f}, rho={metrics_b['Spearman_rho']:.3f}, "
              f"kaimynai_15 su A={row['kaimynai_15']:.3f} ({time.time() - start:.0f} s)")

    return pd.DataFrame(rows), rho_space, space_overlap


# A ir C
def compare_a_c(data_a, X_a, data_c, X_c, projections_a):
    print("\n=== A ir C (be ekstremalių išskirčių), lyginami tik bendri objektai ===")
    # Bendrų objektų eilutės A aibėje, ta pačia tvarka kaip C aibėje.
    position_in_a = pd.Series(np.arange(len(data_a)), index=data_a["id"])
    common_rows = position_in_a.loc[data_c["id"]].to_numpy()
    print(f"A: {len(data_a)} objektų, C: {len(data_c)}, bendri: {len(common_rows)}")

    labels_c = data_c[CLASS_COLUMN].to_numpy()
    X_a_common = X_a[common_rows]

    rows = []
    for seed in SEEDS:
        start = time.time()
        Y_a_common = projections_a[seed][common_rows]   # 1–2 žingsniai
        Y_c = project(X_c, "C", seed)                   # 3 žingsnis

        # Kokybė tų pačių objektų atžvilgiu (A normavimas ir C normavimas).
        metrics_a = evaluate_projection(X_a_common, Y_a_common, labels_c)
        metrics_c = evaluate_projection(X_c, Y_c, labels_c)

        row = {"seed": seed}
        for metric in LOCAL_METRICS:
            row[f"A_bendri_{metric}"] = metrics_a[metric]
            row[f"C_{metric}"] = metrics_c[metric]
        row["rho_projekciju"] = distance_rho(Y_a_common, Y_c)   # 4 žingsnis
        row.update(overlaps(Y_a_common, Y_c))
        rows.append(row)
        print(f"  sėkla {seed}: C T_15={metrics_c['T_15']:.3f}, rho={metrics_c['Spearman_rho']:.3f}, "
              f"kaimynai_15 su A={row['kaimynai_15']:.3f} ({time.time() - start:.0f} s)")

    return pd.DataFrame(rows), common_rows


# Suvestinės
def mean_std(values):
    return f"{values.mean():.3f} ± {values.std():.3f}"


def quality_table(runs, first, second):
    table = pd.DataFrame({
        first: [mean_std(runs[f"{first}_{metric}"]) for metric in LOCAL_METRICS],
        second: [mean_std(runs[f"{second}_{metric}"]) for metric in LOCAL_METRICS],
        "pokytis": [f"{(runs[f'{second}_{metric}'] - runs[f'{first}_{metric}']).mean():+.3f}"
                    for metric in LOCAL_METRICS],
    }, index=LOCAL_METRICS)
    return table


def similarity_table(runs, baseline, name):
    columns = ["rho_projekciju"] + [f"kaimynai_{k}" for k in OVERLAP_K]
    return pd.DataFrame({
        name: [mean_std(runs[column]) for column in columns],
        "A sėkla prieš A sėklą": [f"{baseline[column]:.3f}" for column in columns],
    }, index=columns)


def save_and_print(table, title, file_name):
    print(f"\n{title}:")
    print(table.to_string())
    table.to_csv(COMPARE_DIR / file_name, encoding="utf-8-sig")


def plot_examples(data_b, X_b, data_c, X_c):
    # Po vieną B ir C projekciją (sėkla 0) atskiruose paveiksluose.
    for name, data, X in [("B", data_b, X_b), ("C", data_c, X_c)]:
        Y = project(X, name, 0)
        metrics = evaluate_projection(X, Y, data[CLASS_COLUMN].to_numpy())
        title = (f"{name} aibė. " + parameters_title(**CHOSEN) + ", sėkla 0\n"
                 + metric_line(metrics))
        plot_projection(Y, data, title, FIGURES_DIR / f"umap_{name}_{CHOSEN_NAME}_seed0.png")
    print(f"\nPaveikslai: {FIGURES_DIR}")


def main():
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("UMAP projekcijos jautrumas duomenų paruošimui")
    print(f"Parametrai: {CHOSEN}, sėklos {SEEDS}")

    data_a, X_a = load_subset("A")
    data_b, X_b = load_subset("B")
    data_c, X_c = load_subset("C")
    projections_a = {seed: project_a(seed) for seed in SEEDS}

    runs_b, rho_space, space_overlap = compare_a_b(data_a, X_a, X_b, projections_a)
    runs_c, common_rows = compare_a_c(data_a, X_a, data_c, X_c, projections_a)

    runs_b.round(4).to_csv(COMPARE_DIR / "A_B_paleidimai.csv", index=False, encoding="utf-8-sig")
    runs_c.round(4).to_csv(COMPARE_DIR / "A_C_paleidimai.csv", index=False, encoding="utf-8-sig")

    print("\nSkaičiuojamas atskaitos taškas (A sėklos tarpusavyje)...")
    baseline_all = seed_baseline(projections_a)
    baseline_common = seed_baseline(projections_a, common_rows)

    save_and_print(quality_table(runs_b, "A", "B"),
                   "A ir B projekcijų kokybė (vidurkis ± std, 5 sėklos)", "A_B_kokybe.csv")
    save_and_print(similarity_table(runs_b, baseline_all, "A prieš B (ta pati sėkla)"),
                   f"A ir B projekcijų panašumas; pradinės erdvės rho(D_A, D_B) = {rho_space:.3f}",
                   "A_B_panasumas.csv")
    save_and_print(quality_table(runs_c, "A_bendri", "C"),
                   "A (tik bendri objektai) ir C projekcijų kokybė", "A_C_kokybe.csv")
    save_and_print(similarity_table(runs_c, baseline_common, "A bendri prieš C (ta pati sėkla)"),
                   "A ir C projekcijų panašumas (bendri objektai)", "A_C_panasumas.csv")

    plot_examples(data_b, X_b, data_c, X_c)
    print(f"Lentelės: {COMPARE_DIR}")


if __name__ == "__main__":
    main()