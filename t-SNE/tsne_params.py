# 12 uždavinys: t-SNE parametrų tyrimas su A aibe.
# Tiriama:
#   perplexity = 5, 15, 30, 50 – efektyvus kaimynų skaičius
#                                (užduotyje rekomenduojamas intervalas 5–50);
#   init       = pca, random   – pradinių koordinačių generavimas.
# Kiti parametrai – scikit-learn numatytieji (learning_rate = "auto",
# early_exaggeration = 12, max_iter = 1000).
# Visi deriniai skaičiuojami su ta pačia sėkla, kad skirtumus lemtų tik
# parametrai. Sėklų poveikis tiriamas atskirai (tsne_stability.py).
#
# Klasės stulpelis t-SNE NEPADUODAMAS – jis naudojamas tik NH ir kNN tikslumui.
#
# Įvestis: data/A_norm.csv (sukuria scaling.py)
# Išvestis: rezultatai/tsne/tsne_parametrai.csv ir projekcijos .npy failai
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import evaluate_projection

INPUT_FILE = ROOT / "data" / "A_norm.csv"
OUTPUT_DIR = ROOT / "rezultatai" / "tsne"
RESULTS_FILE = OUTPUT_DIR / "tsne_parametrai.csv"
PROJECTIONS_DIR = OUTPUT_DIR / "projekcijos"

CLASS_COLUMN = "klase"
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", "ekstremali_isskirtis"]

PERPLEXITIES = [5, 15, 30, 50]
INITS = ["pca", "random"]
N_COMPONENTS = 2
RANDOM_STATE = 42

# scikit-learn numatytieji parametrai – atskaitos taškas palyginimui.
DEFAULT = {"perplexity": 30, "init": "pca"}


def load_data(path=INPUT_FILE):
    data = pd.read_csv(path)
    features = [column for column in data.columns if column not in ID_COLUMNS]
    return data, data[features].to_numpy(), data[CLASS_COLUMN].to_numpy(), features


def run_tsne(X, perplexity, init, random_state):
    model = TSNE(
        n_components=N_COMPONENTS,
        perplexity=perplexity,
        init=init,
        random_state=random_state,
    )
    return model.fit_transform(X)


def projection_file(perplexity, init):
    return PROJECTIONS_DIR / f"tsne_p{perplexity}_{init}.npy"


def run_grid(X, labels):
    rows = []
    combinations = list(product(PERPLEXITIES, INITS))

    for number, (perplexity, init) in enumerate(combinations, start=1):
        start = time.time()
        Y = run_tsne(X, perplexity, init, RANDOM_STATE)
        np.save(projection_file(perplexity, init), Y)

        metrics = evaluate_projection(X, Y, labels)
        rows.append({
            "perplexity": perplexity,
            "init": init,
            "numatytieji": (perplexity, init) == tuple(DEFAULT.values()),
            **metrics,
        })
        print(f"[{number}/{len(combinations)}] perplexity={perplexity}, init={init}: "
              f"T_15={metrics['T_15']:.3f}, C_15={metrics['C_15']:.3f}, "
              f"rho={metrics['Spearman_rho']:.3f}, NH={metrics['NH_10']:.3f} "
              f"({time.time() - start:.0f} s)")

    return pd.DataFrame(rows)


def print_summary(results):
    shown = ["perplexity", "init", "T_5", "T_15", "T_50", "C_15", "kNN_15",
             "Spearman_rho", "Stress_1", "NH_10", "kNN_tikslumas"]

    print("\nVisi deriniai:")
    print(results[shown].round(3).to_string(index=False))

    print("\nNumatytieji parametrai (perplexity=30, init=pca):")
    print(results.loc[results["numatytieji"], shown].round(3).to_string(index=False))

    print("\nGeriausi deriniai pagal kiekvieną metriką:")
    for metric, best_is_max in [("T_15", True), ("C_15", True), ("kNN_15", True),
                                ("Spearman_rho", True), ("Stress_1", False),
                                ("NH_10", True)]:
        best = results.loc[results[metric].idxmax() if best_is_max else results[metric].idxmin()]
        print(f"  {metric:<13} {best[metric]:.3f}  "
              f"(perplexity={best['perplexity']}, init={best['init']})")

    print("\nVidurkis pagal perplexity (per abi inicializacijas):")
    print(results.groupby("perplexity")[["T_5", "T_15", "T_50", "C_15",
                                         "Spearman_rho", "NH_10"]].mean().round(3).to_string())

    print("\nVidurkis pagal init:")
    print(results.groupby("init")[["T_15", "C_15", "Spearman_rho", "NH_10"]]
          .mean().round(3).to_string())


def main():
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("12 uždavinys: t-SNE parametrų tyrimas\n")
    _, X, labels, features = load_data()
    print(f"A aibė: {X.shape[0]} objektų, {X.shape[1]} požymiai: {', '.join(features)}")
    print(f"Derinių: {len(PERPLEXITIES) * len(INITS)}, random_state = {RANDOM_STATE}\n")

    results = run_grid(X, labels)
    results.round(4).to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    print_summary(results)
    print(f"\nIšsaugota: {RESULTS_FILE}")
    print(f"Projekcijos: {PROJECTIONS_DIR}")


if __name__ == "__main__":
    main()
