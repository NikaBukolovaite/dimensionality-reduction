# geriausių parametrų pasirinkimas ir stabilumas.

# umap_params.py kiekvienas derinys skaičiuotas su viena sėkla.
# Kandidatų skirtumai ten buvo maži (pvz., Spearman rho 0,68–0,71), todėl
# neaišku, ar juos lemia parametrai, ar atsitiktinumas. Čia kiekvienas
# kandidatas paleidžiamas su 5 skirtingomis sėklomis ir pateikiamas
# vidurkis ± standartinis nuokrypis (kaip reikalaujama skaidrėse).

# Klasės stulpelis UMAP NEPADUODAMAS – jis naudojamas tik NH ir kNN tikslumui.

# Įvestis: data/A_norm.csv (sukuria scaling.py)
# Išvestis: rezultatai/umap/umap_stabilumas.csv (visi paleidimai)
#           rezultatai/umap/umap_stabilumas_suvestine.csv (vidurkis ± std)

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import umap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics import evaluate_projection

warnings.filterwarnings("ignore", message="n_jobs value")

INPUT_FILE = ROOT / "data" / "A_norm.csv"
OUTPUT_DIR = ROOT / "rezultatai" / "umap"
RUNS_FILE = OUTPUT_DIR / "umap_stabilumas.csv"
SUMMARY_FILE = OUTPUT_DIR / "umap_stabilumas_suvestine.csv"
PROJECTIONS_DIR = OUTPUT_DIR / "stabilumas"

CLASS_COLUMN = "klase"
ID_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", "ekstremali_isskirtis"]

# Kandidatai iš 13 uždavinio: numatytieji ir du su daugiau kaimynų
# (geresnė globali struktūra). Visi su min_dist = 0.1 ir spektrine
# inicializacija, nes 0.5 blogino lokalią struktūrą, o random – globalią.
CANDIDATES = [
    {"n_neighbors": 15, "min_dist": 0.1, "init": "spectral"},
    {"n_neighbors": 30, "min_dist": 0.1, "init": "spectral"},
    {"n_neighbors": 50, "min_dist": 0.1, "init": "spectral"},
]
SEEDS = [0, 1, 2, 3, 4]

SHOWN_METRICS = ["T_5", "T_15", "T_50", "C_15", "kNN_15",
                 "Spearman_rho", "Stress_1", "NH_10", "kNN_tikslumas"]


def load_data():
    data = pd.read_csv(INPUT_FILE)
    features = [column for column in data.columns if column not in ID_COLUMNS]
    return data[features].to_numpy(), data[CLASS_COLUMN].to_numpy()


def candidate_name(candidate):
    return f"nn{candidate['n_neighbors']}_md{candidate['min_dist']}_{candidate['init']}"


def run_all(X, labels):
    rows = []
    total = len(CANDIDATES) * len(SEEDS)
    number = 0

    for candidate in CANDIDATES:
        for seed in SEEDS:
            number += 1
            start = time.time()
            Y = umap.UMAP(n_components=2, random_state=seed, **candidate).fit_transform(X)
            np.save(PROJECTIONS_DIR / f"umap_{candidate_name(candidate)}_seed{seed}.npy", Y)

            metrics = evaluate_projection(X, Y, labels)
            rows.append({**candidate, "seed": seed, **metrics})
            print(f"[{number}/{total}] {candidate_name(candidate)}, seed={seed}: "
                  f"T_15={metrics['T_15']:.3f}, rho={metrics['Spearman_rho']:.3f}, "
                  f"NH={metrics['NH_10']:.3f} ({time.time() - start:.0f} s)")

    return pd.DataFrame(rows)


def summarize(runs):
    grouped = runs.groupby(["n_neighbors", "min_dist", "init"])[SHOWN_METRICS]
    mean = grouped.mean()
    std = grouped.std()

    # Lentelė ataskaitai: "vidurkis ± std" kiekvienai metrikai.
    text = mean.map(lambda v: f"{v:.3f}") + " ± " + std.map(lambda v: f"{v:.3f}")
    return mean, std, text


def print_summary(mean, std, text):
    print("\nVidurkis ± standartinis nuokrypis (5 sėklos):")
    print(text.T.to_string())

    print("\nDidžiausias standartinis nuokrypis kiekvienai metrikai "
          "(kiek rezultatas priklauso nuo sėklos):")
    print(std.max().round(3).to_string())

    print("\nAr skirtumai tarp kandidatų didesni už sėklų sklaidą?")
    for metric in ["T_15", "C_15", "Spearman_rho", "NH_10"]:
        difference = mean[metric].max() - mean[metric].min()
        spread = std[metric].max()
        verdict = "taip" if difference > 2 * spread else "ne"
        print(f"  {metric:<13} skirtumas {difference:.3f}, sklaida {spread:.3f} -> {verdict}")


def main():
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("14–15 uždaviniai: UMAP kandidatų stabilumas\n")
    X, labels = load_data()
    print(f"A aibė: {X.shape[0]} objektų; kandidatų {len(CANDIDATES)}, sėklos {SEEDS}\n")

    runs = run_all(X, labels)
    runs.round(4).to_csv(RUNS_FILE, index=False, encoding="utf-8-sig")

    mean, std, text = summarize(runs)
    print_summary(mean, std, text)
    text.to_csv(SUMMARY_FILE, encoding="utf-8-sig")

    print(f"\nIšsaugota: {RUNS_FILE}")
    print(f"Išsaugota: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()