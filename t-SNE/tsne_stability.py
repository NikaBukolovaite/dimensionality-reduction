# 14–15 uždaviniai (t-SNE dalis): geriausių parametrų pasirinkimas ir stabilumas.
#
# 12 uždavinyje (tsne_params.py) kiekvienas derinys skaičiuotas su viena sėkla.
# Čia geriausi kandidatai (perplexity = 15, 30, 50) paleidžiami su 5 sėklomis:
#   - su atsitiktine inicializacija – kiek rezultatas priklauso nuo sėklos
#     (vidurkis ± standartinis nuokrypis, kaip reikalaujama skaidrėse);
#   - su PCA inicializacija – dviem sėklomis patikrinama, ar sėkla apskritai keičia rezultatą.
#     scikit-learn t-SNE sėklą naudoja tik atsitiktinėms pradinėms
#     koordinatėms, todėl su PCA inicializacija rezultatas turėtų sutapti.
#
# Klasės stulpelis t-SNE NEPADUODAMAS – jis naudojamas tik NH ir kNN tikslumui.
#
# Įvestis: data/A_norm.csv
# Išvestis: rezultatai/tsne/tsne_stabilumas.csv (visi paleidimai)
#           rezultatai/tsne/tsne_stabilumas_suvestine.csv (vidurkis ± std)
import warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TSNE"))

from metrics import evaluate_projection
from tsne_params import load_data, run_tsne

OUTPUT_DIR = ROOT / "rezultatai" / "tsne"
RUNS_FILE = OUTPUT_DIR / "tsne_stabilumas.csv"
SUMMARY_FILE = OUTPUT_DIR / "tsne_stabilumas_suvestine.csv"
PROJECTIONS_DIR = OUTPUT_DIR / "stabilumas"

PERPLEXITIES = [15, 30, 50]
INITS = ["pca", "random"]
SEEDS = [0, 1, 2, 3, 4]
# Su PCA inicializacija sėkla rezultato nekeičia, todėl patikrai užtenka dviejų sėklų.
PCA_SEEDS = [0, 1]

SHOWN_METRICS = ["T_5", "T_15", "T_50", "C_15", "kNN_15",
                 "Spearman_rho", "Stress_1", "NH_10", "kNN_tikslumas"]


def projection_file(perplexity, init, seed):
    return PROJECTIONS_DIR / f"tsne_p{perplexity}_{init}_seed{seed}.npy"


def run_all(X, labels):
    rows = []
    total = len(PERPLEXITIES) * (len(SEEDS) + len(PCA_SEEDS))
    number = 0

    for perplexity in PERPLEXITIES:
        for init in INITS:
            first_projection = None
            for seed in (PCA_SEEDS if init == "pca" else SEEDS):
                number += 1
                start = time.time()
                Y = run_tsne(X, perplexity, init, seed)
                np.save(projection_file(perplexity, init, seed), Y)

                # Kiek projekcija skiriasi nuo tos pačios konfigūracijos su pirma sėkla.
                if first_projection is None:
                    first_projection = Y
                difference = float(np.abs(Y - first_projection).max())

                metrics = evaluate_projection(X, Y, labels)
                rows.append({"perplexity": perplexity, "init": init, "seed": seed,
                             "max_skirtumas_nuo_seed0": difference, **metrics})
                print(f"[{number}/{total}] perplexity={perplexity}, init={init}, seed={seed}: "
                      f"T_15={metrics['T_15']:.3f}, rho={metrics['Spearman_rho']:.3f}, "
                      f"NH={metrics['NH_10']:.3f}, skirtumas nuo seed 0 = {difference:.3g} "
                      f"({time.time() - start:.0f} s)")

    return pd.DataFrame(rows)


def summarize(runs):
    grouped = runs.groupby(["init", "perplexity"])[SHOWN_METRICS]
    mean = grouped.mean()
    std = grouped.std()
    text = mean.map(lambda v: f"{v:.3f}") + " ± " + std.map(lambda v: f"{v:.3f}")
    return mean, std, text


def print_summary(runs, mean, std, text):
    print("\nVidurkis ± standartinis nuokrypis (atsitiktinė init. – 5 sėklos, PCA init. – 2 sėklos):")
    print(text.T.to_string())

    print("\nAr sėkla keičia projekciją? (didžiausias koordinačių skirtumas nuo sėklos 0)")
    print(runs.groupby(["init", "perplexity"])["max_skirtumas_nuo_seed0"].max()
          .map(lambda v: f"{v:.3g}").to_string())

    print("\nAr skirtumai tarp perplexity didesni už sėklų sklaidą (atsitiktinė inicializacija)?")
    random_mean, random_std = mean.loc["random"], std.loc["random"]
    for metric in ["T_15", "C_15", "Spearman_rho", "NH_10"]:
        difference = random_mean[metric].max() - random_mean[metric].min()
        spread = random_std[metric].max()
        verdict = "taip" if difference > 2 * spread else "ne"
        print(f"  {metric:<13} skirtumas {difference:.3f}, sklaida {spread:.3f} -> {verdict}")


def main():
    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("14–15 uždaviniai: t-SNE kandidatų stabilumas\n")
    _, X, labels, _ = load_data()
    print(f"A aibė: {X.shape[0]} objektų; perplexity {PERPLEXITIES}, "
          f"inicializacijos {INITS}, sėklos {SEEDS}\n")

    runs = run_all(X, labels)
    runs.round(4).to_csv(RUNS_FILE, index=False, encoding="utf-8-sig")

    mean, std, text = summarize(runs)
    print_summary(runs, mean, std, text)
    text.to_csv(SUMMARY_FILE, encoding="utf-8-sig")

    print(f"\nIšsaugota: {RUNS_FILE}")
    print(f"Išsaugota: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
