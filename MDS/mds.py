"ka darom: naudojam A aibe, atliekam MDS (SMACOF) metrini ir nemetrini variantus, ivertinam abu (metrics.py), issaugom lenteles ir grafikus i rezultatai/mds/A/. Pradines koordinates imam is PCA, kad rezultatas butu stabilus (nemetrinis MDS su atsitiktine pradzia nesukonverguoja). Palyginimui abu variantai paleidziami ir su numatytaja atsitiktine pradzia."

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PCA"))
os.chdir(ROOT)
#used for mac
warnings.filterwarnings("ignore", message=".*encountered in matmul")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

from metrics import evaluate_projection
from pca import fit_pca, project


INPUT_FILE = Path("data/A_norm.csv")
OUTPUT_DIR = Path("rezultatai/mds/A")

FEATURES = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]
CLASSES = ["N", "L", "R", "V", "A"]
COLOURS = {"N": "#2a78d6", "L": "#eb6834", "R": "#1baf7a", "V": "#eda100", "A": "#e87ba4"}

VARIANTS = ["metrinis", "nemetrinis"]   # naudojami ir mds_AB.py, mds_AC.py
STARTS = ["PCA", "atsitiktinė"]         # pradinės koordinatės (tik A aibėje)
MAX_ITER = 300
RANDOM_STATE = 0


def run_mds(X, variant, start="PCA"):
    model = MDS(
        n_components=2,
        metric=(variant == "metrinis"),
        n_init=1,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        dissimilarity="euclidean",
    )

    # atsitiktinė pradžia – sklearn numatytoji
    if start == "atsitiktinė":
        return model.fit_transform(X)

    # pradinės koordinatės – PCA 2D projekcija (vienodos abiem variantams)
    mean, eigenvalues, eigenvectors = fit_pca(X)
    initial = project(X, mean, eigenvectors, 2)
    return model.fit_transform(X, init=initial)


def plot_projection(Y, data, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for cls in CLASSES:
        normal = (data["klase"] == cls) & ~data["ekstremali_isskirtis"]
        outlier = (data["klase"] == cls) & data["ekstremali_isskirtis"]
        ax.scatter(Y[normal, 0], Y[normal, 1], s=7, alpha=0.5, color=COLOURS[cls])
        ax.scatter(Y[outlier, 0], Y[outlier, 1], s=20, color=COLOURS[cls], edgecolors="black")

    # legenda: klasės + išskirties žymėjimas (tik jei aibėje yra išskirčių, C aibėje jų nėra)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=COLOURS[cls], label=cls) for cls in CLASSES]
    if data["ekstremali_isskirtis"].any():
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                                  markerfacecolor="white", markeredgecolor="black",
                                  label="ekstremali išskirtis"))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    ax.legend(handles=handles, title="Klasė", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)
    X = data[FEATURES].to_numpy()
    labels = data["klase"].to_numpy()
    print(f"A aibė: {X.shape[0]} objektų, {X.shape[1]} požymiai\n")

    metrics = {}
    for variant in VARIANTS:
        for start in STARTS:
            print(f"(skaiciuoja {variant} MDS, {start} pradžia, ~1 min)")
            Y = run_mds(X, variant, start)
            metrics[f"{variant} ({start})"] = evaluate_projection(X, Y, labels)

            # projekcijos ir grafikai – tik pagrindinei (PCA) pradžiai
            if start == "PCA":
                np.save(OUTPUT_DIR / f"mds_A_{variant}.npy", Y)
                plot_projection(Y, data, f"MDS ({variant}): A aibės projekcija",
                                OUTPUT_DIR / f"mds_A_{variant}.png")

    metrics = pd.DataFrame(metrics)
    print("\nProjekcijos kokybė:")
    print(metrics.round(3), "\n")
    metrics.round(4).to_csv(OUTPUT_DIR / "mds_metrikos.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()