"ka darom: naudojam A aibe, atliekam PCA (centravimas -> kovariacija -> eigenvektoriai), tikrinam ar rezultatas sutampa su sklearn PCA, isvedam paaiskinta dispersija ir pozymiu svorius, ivertinam projekcijos kokybe 2D ir 3D (metrics.py), issaugom lenteles ir du grafikus i rezultatai/pca/A/"

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
#used for mac
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from metrics import evaluate_projection


INPUT_FILE = Path("data/A_norm.csv")
OUTPUT_DIR = Path("rezultatai/pca/A")

FEATURES = ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"]
CLASSES = ["N", "L", "R", "V", "A"]
COLOURS = {"N": "#2a78d6", "L": "#eb6834", "R": "#1baf7a", "V": "#eda100", "A": "#e87ba4"}


def fit_pca(X):
    mean = X.mean(axis=0)
    X_centered = X - mean

    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for i in range(eigenvectors.shape[1]):
        if eigenvectors[np.abs(eigenvectors[:, i]).argmax(), i] < 0:
            eigenvectors[:, i] *= -1
    return mean, eigenvalues, eigenvectors


def project(X, mean, eigenvectors, n_components):
    return (X - mean) @ eigenvectors[:, :n_components]


#for checking purposes
def check_with_sklearn(X, eigenvalues, eigenvectors):
    sk = SklearnPCA().fit(X)
    same_variance = np.allclose(eigenvalues / eigenvalues.sum(), sk.explained_variance_ratio_)
    same_vectors = np.allclose(np.abs(eigenvectors.T), np.abs(sk.components_))
    assert same_variance and same_vectors, "nesutampa su sklearn PCA"



def plot_variance(variance_percent):
    pcs = [f"PC{i + 1}" for i in range(len(variance_percent))]
    cumulative = np.cumsum(variance_percent)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(pcs, variance_percent, color="#2a78d6", label="komponentės dispersija")
    ax.plot(pcs, cumulative, color="black", marker="o", label="sukaupta dispersija")
    for x, y in zip(pcs, cumulative):
        ax.annotate(f"{y:.1f} %", (x, y), textcoords="offset points", xytext=(0, 7), ha="center")

    ax.set_ylabel("Išsaugota dispersija, %")
    ax.set_ylim(0, 105)
    ax.set_title("Išsaugota dispersija pagal komponenčių skaičių")
    ax.legend(frameon=False, loc="center right")

    fig.savefig(OUTPUT_DIR / "pca_dispersija.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_projection(Y, data, variance_percent):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # išskirtys su juodu kraštu
    for cls in CLASSES:
        normal = (data["klase"] == cls) & ~data["ekstremali_isskirtis"]
        outlier = (data["klase"] == cls) & data["ekstremali_isskirtis"]
        ax.scatter(Y[normal, 0], Y[normal, 1], s=7, alpha=0.5, color=COLOURS[cls], label=cls)
        ax.scatter(Y[outlier, 0], Y[outlier, 1], s=20, color=COLOURS[cls], edgecolors="black")

    ax.set_aspect("equal", adjustable="datalim")  
    ax.set_xlabel(f"PC1 ({variance_percent[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({variance_percent[1]:.1f} %)")
    ax.set_title("PCA: A aibės projekcija")
    ax.legend(title="Klasė", markerscale=2, frameon=False)

    fig.savefig(OUTPUT_DIR / "pca_A_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_FILE)
    X = data[FEATURES].to_numpy()
    labels = data["klase"].to_numpy()
    print(f"A aibė: {X.shape[0]} objektų, {X.shape[1]} požymiai\n")

    mean, eigenvalues, eigenvectors = fit_pca(X)
    check_with_sklearn(X, eigenvalues, eigenvectors)

    pcs = [f"PC{i + 1}" for i in range(len(FEATURES))]
    variance_percent = eigenvalues / eigenvalues.sum() * 100
    variance = pd.DataFrame({
        "dispersija_%": variance_percent,
        "sukaupta_%": np.cumsum(variance_percent),
    }, index=pcs)
    print("Paaiškinta dispersija:")
    print(variance.round(1), "\n")

    loadings = pd.DataFrame(eigenvectors, index=FEATURES, columns=pcs)
    print("Požymių svoriai:")
    print(loadings.round(3), "\n")
    print("(skaiciuoja)")
    metrics = {}
    for d in [2, 3]:
        Y = project(X, mean, eigenvectors, d)
        metrics[f"{d}D"] = evaluate_projection(X, Y, labels)
    metrics = pd.DataFrame(metrics)
    print("\nProjekcijos kokybė:")
    print(metrics.round(3), "\n")

    Y_2d = project(X, mean, eigenvectors, 2)
    np.save(OUTPUT_DIR / "pca_A_2d.npy", Y_2d)
    variance.round(4).to_csv(OUTPUT_DIR / "pca_dispersija.csv", encoding="utf-8-sig")
    loadings.round(4).to_csv(OUTPUT_DIR / "pca_svoriai.csv", encoding="utf-8-sig")
    metrics.round(4).to_csv(OUTPUT_DIR / "pca_metrikos.csv", encoding="utf-8-sig")
    plot_variance(variance_percent)
    plot_projection(Y_2d, data, variance_percent)


if __name__ == "__main__":
    main()