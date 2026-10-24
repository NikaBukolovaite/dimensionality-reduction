"ka darom: PCA A ir C aibems (C = A be ekstremaliu isskirciu), lyginam ar isskirtys keite likusiu objektu struktura. Objektu skaicius skiriasi, todel A projekcijoje paliekam tik bendrus objektus (pagal id)."

from pca import fit_pca, project, CLASSES, COLOURS, FEATURES

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from metrics import evaluate_projection, neighbor_overlap, K_VALUES


FILE_A = Path("data/A_norm.csv")
FILE_C = Path("data/C_norm.csv")
OUTPUT_DIR = Path("rezultatai/pca/AC")


def plot_projection(Y, data, variance_percent, title, file_name):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for cls in CLASSES:
        in_class = data["klase"] == cls
        ax.scatter(Y[in_class, 0], Y[in_class, 1], s=7, alpha=0.5, color=COLOURS[cls], label=cls)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"PC1 ({variance_percent[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({variance_percent[1]:.1f} %)")
    ax.set_title(title)
    ax.legend(title="Klasė", markerscale=2, frameon=False)

    fig.savefig(OUTPUT_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_a = pd.read_csv(FILE_A)
    data_c = pd.read_csv(FILE_C)

    X_a = data_a[FEATURES].to_numpy()
    X_c = data_c[FEATURES].to_numpy()
    labels_c = data_c["klase"].to_numpy()
    print(f"A: {X_a.shape[0]} objektų")
    print(f"C: {X_c.shape[0]} objektų (pašalinta {X_a.shape[0] - X_c.shape[0]} išskirčių)\n")

    mean_a, eigenvalues_a, eigenvectors_a = fit_pca(X_a)
    Y_a = project(X_a, mean_a, eigenvectors_a, 2)

    #paliekam tik bendrus objektus (tie, kurie yra ir C)
    common = data_a["id"].isin(data_c["id"]).to_numpy()
    assert (data_a.loc[common, "id"].to_numpy() == data_c["id"].to_numpy()).all(), \
        "bendrų objektų tvarka A ir C turi sutapti"
    X_a_common = X_a[common]
    Y_a_common = Y_a[common]

    # PCA iš naujo be išskirčių 
    mean_c, eigenvalues_c, eigenvectors_c = fit_pca(X_c)
    Y_c = project(X_c, mean_c, eigenvectors_c, 2)

    variance_c = eigenvalues_c / eigenvalues_c.sum() * 100

    pcs = [f"PC{i + 1}" for i in range(len(FEATURES))]
    variance = pd.DataFrame({
        "dispersija_%": variance_c,
        "sukaupta_%": np.cumsum(variance_c),
    }, index=pcs)
    loadings = pd.DataFrame(eigenvectors_c, index=FEATURES, columns=pcs)
    print("C: paaiškinta dispersija:")
    print(variance.round(1), "\n")
    print("C: požymių svoriai:")
    print(loadings.round(3), "\n")

    #kokybė tik bendriems objektams
    print("(skaiciuoja)")
    metrics = pd.DataFrame({
        "A (bendri objektai)": evaluate_projection(X_a_common, Y_a_common, labels_c),
        "C": evaluate_projection(X_c, Y_c, labels_c),
    })
    metrics["pokytis"] = metrics["C"] - metrics["A (bendri objektai)"]
    print("\nProjekcijos kokybė, A ir C (bendri objektai):")
    print(metrics.round(3), "\n")

    #bendrų objektų atstumai ir kaimynystes A ir C projekcijose
    comparison = {"rho(Y_A, Y_C) bendri objektai": spearmanr(pdist(Y_a_common), pdist(Y_c))[0]}
    for k in K_VALUES:
        comparison[f"kaimynų sutapimas Y, k={k}"] = neighbor_overlap(Y_a_common, Y_c, k)
    comparison = pd.Series(comparison, name="reikšmė")
    print("A ir C palyginimas:")
    print(comparison.round(3), "\n")

    np.save(OUTPUT_DIR / "pca_C_2d.npy", Y_c)
    variance.round(4).to_csv(OUTPUT_DIR / "pca_C_dispersija.csv", encoding="utf-8-sig")
    loadings.round(4).to_csv(OUTPUT_DIR / "pca_C_svoriai.csv", encoding="utf-8-sig")
    metrics.round(4).to_csv(OUTPUT_DIR / "pca_AC_metrikos.csv", encoding="utf-8-sig")
    comparison.round(4).to_csv(OUTPUT_DIR / "pca_AC_palyginimas.csv", encoding="utf-8-sig")
    plot_projection(Y_c, data_c, variance_c, "PCA: C aibės projekcija (be išskirčių)", "pca_C_2d.png")


if __name__ == "__main__":
    main()