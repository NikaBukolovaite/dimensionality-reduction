"ka darom: PCA A ir B aibems (B = A be RR_sant_post), lyginam ar pasalinus koreliuota pozymi isliko ta pati objektu struktura. Objektai A ir B tie patys, todel lyginam tiesiogiai."

from pca import fit_pca, project, CLASSES, COLOURS, FEATURES

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from metrics import evaluate_projection, neighbor_overlap, K_VALUES


FILE_A = Path("data/A_norm.csv")
FILE_B = Path("data/B_norm.csv")
OUTPUT_DIR = Path("rezultatai/pca/AB")

FEATURES_A = FEATURES
FEATURES_B = [f for f in FEATURES if f != "RR_sant_post"]


def plot_projection(Y, data, variance_percent, title, file_name):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # išskirtys su juodu kraštu
    for cls in CLASSES:
        normal = (data["klase"] == cls) & ~data["ekstremali_isskirtis"]
        outlier = (data["klase"] == cls) & data["ekstremali_isskirtis"]
        ax.scatter(Y[normal, 0], Y[normal, 1], s=7, alpha=0.5, color=COLOURS[cls])
        ax.scatter(Y[outlier, 0], Y[outlier, 1], s=20, color=COLOURS[cls], edgecolors="black")

    # legenda: klasės + išskirties žymėjimas
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          color=COLOURS[cls], label=cls) for cls in CLASSES]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                              markerfacecolor="white", markeredgecolor="black",
                              label="ekstremali išskirtis"))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({variance_percent[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({variance_percent[1]:.1f} %)")
    ax.set_title(title)
    ax.legend(handles=handles, title="Klasė", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.savefig(OUTPUT_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_a = pd.read_csv(FILE_A)
    data_b = pd.read_csv(FILE_B)
    assert (data_a["id"] == data_b["id"]).all(), "A ir B objektai turi sutapti"

    X_a = data_a[FEATURES_A].to_numpy()
    X_b = data_b[FEATURES_B].to_numpy()
    labels = data_a["klase"].to_numpy()
    print(f"A: {X_a.shape[0]} objektų, {X_a.shape[1]} požymiai")
    print(f"B: {X_b.shape[0]} objektų, {X_b.shape[1]} požymiai\n")

    # PCA atskirai kiekvienai aibei, 2D
    mean_a, eigenvalues_a, eigenvectors_a = fit_pca(X_a)
    mean_b, eigenvalues_b, eigenvectors_b = fit_pca(X_b)
    Y_a = project(X_a, mean_a, eigenvectors_a, 2)
    Y_b = project(X_b, mean_b, eigenvectors_b, 2)

    variance_b = eigenvalues_b / eigenvalues_b.sum() * 100

    # B aibės dispersija ir svoriai (A jau yra pca.py)
    pcs_b = [f"PC{i + 1}" for i in range(len(FEATURES_B))]
    variance = pd.DataFrame({
        "dispersija_%": variance_b,
        "sukaupta_%": np.cumsum(variance_b),
    }, index=pcs_b)
    loadings = pd.DataFrame(eigenvectors_b, index=FEATURES_B, columns=pcs_b)
    print("B: paaiškinta dispersija:")
    print(variance.round(1), "\n")
    print("B: požymių svoriai:")
    print(loadings.round(3), "\n")

    print("(skaiciuoja)")
    metrics = pd.DataFrame({
        "A": evaluate_projection(X_a, Y_a, labels),
        "B": evaluate_projection(X_b, Y_b, labels),
    })
    metrics["pokytis"] = metrics["B"] - metrics["A"]
    print("\nProjekcijos kokybė, A ir B:")
    print(metrics.round(3), "\n")

    comparison = {
        "rho(D_A, D_B) pradinės erdvės": spearmanr(pdist(X_a), pdist(X_b))[0],
        "rho(Y_A, Y_B) projekcijos": spearmanr(pdist(Y_a), pdist(Y_b))[0],
    }
    for k in K_VALUES:
        comparison[f"kaimynų sutapimas X, k={k}"] = neighbor_overlap(X_a, X_b, k)
    for k in K_VALUES:
        comparison[f"kaimynų sutapimas Y, k={k}"] = neighbor_overlap(Y_a, Y_b, k)
    comparison = pd.Series(comparison, name="reikšmė")
    print("A ir B palyginimas:")
    print(comparison.round(3), "\n")

    np.save(OUTPUT_DIR / "pca_B_2d.npy", Y_b)
    variance.round(4).to_csv(OUTPUT_DIR / "pca_B_dispersija.csv", encoding="utf-8-sig")
    loadings.round(4).to_csv(OUTPUT_DIR / "pca_B_svoriai.csv", encoding="utf-8-sig")
    metrics.round(4).to_csv(OUTPUT_DIR / "pca_AB_metrikos.csv", encoding="utf-8-sig")
    comparison.round(4).to_csv(OUTPUT_DIR / "pca_AB_palyginimas.csv", encoding="utf-8-sig")
    plot_projection(Y_b, data_b, variance_b, "PCA: B aibės projekcija (be RR_sant_post)", "pca_B_2d.png")


if __name__ == "__main__":
    main()