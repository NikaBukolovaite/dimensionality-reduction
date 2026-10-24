# UMAP parametrų tyrimas su A aibe.
# Tiriama:
#   n_neighbors = 5, 15, 30, 50 - kiek kaimynų laikoma "vietine" struktūra (mažas - lokali, didelis - globalesnė);
#   min_dist    = 0.0, 0.1, 0.5 - kaip tankiai suspaudžiamos grupės;
#   init        = spectral, random - pradinių koordinačių generavimas.
# Visi deriniai skaičiuojami su ta pačia sėkla, kad skirtumus lemtų tik
# parametrai. Sėklų poveikis tiriamas atskirai (15 uždavinys).
# Klasės stulpelis UMAP NEPADUODAMAS - jis naudojamas tik NH ir kNN tikslumui.
# Įvestis: data/A_norm.csv (sukuria scaling.py)
# Išvestis: rezultatai/umap/umap_parametrai.csv ir projekcijos .npy failai

from itertools import product
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import umap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import evaluate_projection

INPUT_FILE = ROOT / "data" / "A_norm.csv"
OUTPUT_DIR = ROOT / "rezultatai" / "umap"
RESULTS_FILE = OUTPUT_DIR / "umap_parametrai.csv"
PROJECTIONS_DIR = OUTPUT_DIR / "projekcijos"

CLASS_COLUMN = "klase"
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", "ekstremali_isskirtis"]

N_NEIGHBORS = [5, 15, 30, 50]
MIN_DIST = [0.0, 0.1, 0.5]
INITS = ["spectral", "random"]
N_COMPONENTS = 2
RANDOM_STATE = 42

# UMAP numatytieji parametrai – atskaitos taškas palyginimui.
DEFAULT = {"n_neighbors": 15, "min_dist": 0.1, "init": "spectral"}


def load_data():
    data = pd.read_csv(INPUT_FILE)
    features = [column for column in data.columns if column not in ID_COLUMNS]
    X = data[features].to_numpy()
    labels = data[CLASS_COLUMN].to_numpy()
    print(f"A aibė: {X.shape[0]} objektų, {X.shape[1]} požymiai: {', '.join(features)}")
    return X, labels


def run_umap(X, n_neighbors, min_dist, init):
    model = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init=init,
        random_state=RANDOM_STATE,
    )
    return model.fit_transform(X)


def projection_file(n_neighbors, min_dist, init):
    return PROJECTIONS_DIR / f"umap_nn{n_neighbors}_md{min_dist}_{init}.npy"


def run_grid(X, labels):
    rows = []
    combinations = list(product(N_NEIGHBORS, MIN_DIST, INITS))

    for number, (n_neighbors, min_dist, init) in enumerate(combinations, start=1):
        start = time.time()
        Y = run_umap(X, n_neighbors, min_dist, init)
        np.save(projection_file(n_neighbors, min_dist, init), Y)

        metrics = evaluate_projection(X, Y, labels)
        rows.append({
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "init": init,
            "numatytieji": (n_neighbors, min_dist, init) == tuple(DEFAULT.values()),
            **metrics,
        })
        print(f"[{number}/{len(combinations)}] n_neighbors={n_neighbors}, "
              f"min_dist={min_dist}, init={init}: "
              f"T_15={metrics['T_15']:.3f}, C_15={metrics['C_15']:.3f}, "
              f"rho={metrics['Spearman_rho']:.3f} ({time.time() - start:.0f} s)")

    return pd.DataFrame(rows)


def print_summary(results):
    shown = ["n_neighbors", "min_dist", "init",
             "T_5", "T_15", "T_50", "C_15", "kNN_15",
             "Spearman_rho", "Stress_1", "NH_10", "kNN_tikslumas"]

    print("\nVisi deriniai:")
    print(results[shown].round(3).to_string(index=False))

    print("\nNumatytieji parametrai (n_neighbors=15, min_dist=0.1, spectral):")
    print(results.loc[results["numatytieji"], shown].round(3).to_string(index=False))

    print("\nGeriausi deriniai pagal kiekvieną metriką:")
    for metric, best_is_max in [("T_15", True), ("C_15", True), ("kNN_15", True),
                                ("Spearman_rho", True), ("Stress_1", False),
                                ("NH_10", True)]:
        best = results.loc[results[metric].idxmax() if best_is_max else results[metric].idxmin()]
        print(f"  {metric:<13} {best[metric]:.3f}  "
              f"(n_neighbors={best['n_neighbors']}, min_dist={best['min_dist']}, init={best['init']})")

    # Ar struktūra išlieka keičiant kaimynų skaičių (dėstytojos klausimas).
    print("\nVidurkis pagal n_neighbors (per visus min_dist ir init):")
    print(results.groupby("n_neighbors")[["T_5", "T_15", "T_50", "C_15",
                                          "Spearman_rho", "NH_10"]].mean().round(3).to_string())

    print("\nVidurkis pagal min_dist:")
    print(results.groupby("min_dist")[["T_15", "C_15", "Spearman_rho", "NH_10"]]
          .mean().round(3).to_string())

    print("\nVidurkis pagal init:")
    print(results.groupby("init")[["T_15", "C_15", "Spearman_rho", "NH_10"]]
          .mean().round(3).to_string())


def main():
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("13 uždavinys: UMAP parametrų tyrimas\n")
    X, labels = load_data()
    print(f"Derinių: {len(N_NEIGHBORS) * len(MIN_DIST) * len(INITS)}, "
          f"random_state = {RANDOM_STATE}\n")

    results = run_grid(X, labels)
    results.round(4).to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    print_summary(results)
    print(f"\nIšsaugota: {RESULTS_FILE}")
    print(f"Projekcijos: {PROJECTIONS_DIR}")


if __name__ == "__main__":
    main()