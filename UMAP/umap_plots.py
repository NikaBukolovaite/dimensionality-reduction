from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "A_norm.csv"
UMAP_DIR = ROOT / "rezultatai" / "umap"
PARAMS_DIR = UMAP_DIR / "projekcijos"
STABILITY_DIR = UMAP_DIR / "stabilumas"
PARAMS_FILE = UMAP_DIR / "umap_parametrai.csv"
STABILITY_FILE = UMAP_DIR / "umap_stabilumas.csv"

FIGURES_DIR = UMAP_DIR / "grafikai"
PARAMS_FIGURES_DIR = FIGURES_DIR / "parametrai"
SEEDS_FIGURES_DIR = FIGURES_DIR / "sekios"

CLASS_COLUMN = "klase"
OUTLIER_COLUMN = "ekstremali_isskirtis"
CLASSES = ["N", "L", "R", "V", "A"]

CLASS_COLOURS = {
    "N": "#2a78d6",
    "L": "#eb6834",
    "R": "#1baf7a",
    "V": "#eda100",
    "A": "#e87ba4",
}
INK = "#2b2b2b"

INIT_NAMES = {"spectral": "spektrinė", "random": "atsitiktinė"}
CHOSEN = {"n_neighbors": 50, "min_dist": 0.1, "init": "spectral"}


def comma(value, decimals=3):
    return f"{value:.{decimals}f}".replace(".", ",")


def legend_handles():
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=CLASS_COLOURS[class_name], label=class_name)
        for class_name in CLASSES
    ]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                              markerfacecolor="white", markeredgecolor="black",
                              label="ekstremali išskirtis"))
    return handles


def metric_line(row):
    return (f"T₁₅ = {comma(row['T_15'])}, C₁₅ = {comma(row['C_15'])}, "
            f"ρ = {comma(row['Spearman_rho'])}, NH = {comma(row['NH_10'])}")


def plot_projection(Y, data, title, path):
    is_outlier = data[OUTLIER_COLUMN].to_numpy()
    classes = data[CLASS_COLUMN].to_numpy()

    figure, axis = plt.subplots(figsize=(8.5, 7))

    # Pirmiausia visi įprasti taškai, tada išskirtys ant viršaus.
    for class_name in CLASSES:
        normal = (classes == class_name) & ~is_outlier
        axis.scatter(Y[normal, 0], Y[normal, 1], s=7, alpha=0.55,
                     color=CLASS_COLOURS[class_name], linewidths=0, rasterized=True)
    for class_name in CLASSES:
        outlier = (classes == class_name) & is_outlier
        axis.scatter(Y[outlier, 0], Y[outlier, 1], s=20,
                     color=CLASS_COLOURS[class_name], edgecolors="black",
                     linewidths=0.6, zorder=3, rasterized=True)

    # UMAP ašių reikšmės neturi prasmės, todėl jų nerodome.
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color("#d0d0d0")

    axis.set_title(title, fontsize=12, color=INK)
    axis.legend(handles=legend_handles(), title="Dūžio klasė", loc="upper left",
                bbox_to_anchor=(1.01, 1), frameon=False, fontsize=11)

    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parameters_title(n_neighbors, min_dist, init):
    return (f"UMAP: n_neighbors = {n_neighbors}, min_dist = {comma(min_dist, 1)}, "
            f"{INIT_NAMES[init]} inicializacija")


def plot_all_parameters(data, params):
    for row in params.itertuples(index=False):
        row = row._asdict()
        name = f"umap_nn{row['n_neighbors']}_md{row['min_dist']}_{row['init']}"
        Y = np.load(PARAMS_DIR / f"{name}.npy")
        title = (parameters_title(row["n_neighbors"], row["min_dist"], row["init"])
                 + f"\n{metric_line(row)}")
        plot_projection(Y, data, title, PARAMS_FIGURES_DIR / f"{name}.png")
    print(f"  -> {PARAMS_FIGURES_DIR} ({len(params)} paveikslai)")


def plot_seeds(data, stability):
    chosen = stability[(stability["n_neighbors"] == CHOSEN["n_neighbors"])
                       & (stability["min_dist"] == CHOSEN["min_dist"])
                       & (stability["init"] == CHOSEN["init"])]
    name = f"nn{CHOSEN['n_neighbors']}_md{CHOSEN['min_dist']}_{CHOSEN['init']}"

    for row in chosen.to_dict("records"):
        Y = np.load(STABILITY_DIR / f"umap_{name}_seed{row['seed']}.npy")
        title = (parameters_title(**CHOSEN) + f", sėkla {row['seed']}\n{metric_line(row)}")
        plot_projection(Y, data, title, SEEDS_FIGURES_DIR / f"umap_{name}_seed{row['seed']}.png")
    print(f"  -> {SEEDS_FIGURES_DIR} ({len(chosen)} paveikslai)")


def plot_chosen(data, params):
    row = params[(params["n_neighbors"] == CHOSEN["n_neighbors"])
                 & (params["min_dist"] == CHOSEN["min_dist"])
                 & (params["init"] == CHOSEN["init"])].iloc[0]
    name = f"umap_nn{CHOSEN['n_neighbors']}_md{CHOSEN['min_dist']}_{CHOSEN['init']}"
    Y = np.load(PARAMS_DIR / f"{name}.npy")

    path = FIGURES_DIR / "umap_pasirinkta.png"
    plot_projection(Y, data, parameters_title(**CHOSEN) + f"\n{metric_line(row)}", path)
    print("  ->", path)


def main():
    for folder in [FIGURES_DIR, PARAMS_FIGURES_DIR, SEEDS_FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE)
    params = pd.read_csv(PARAMS_FILE)
    stability = pd.read_csv(STABILITY_FILE)

    print("UMAP grafikai:")
    plot_chosen(data, params)
    plot_all_parameters(data, params)
    plot_seeds(data, stability)


if __name__ == "__main__":
    main()