# 18 uždavinys (t-SNE dalis): t-SNE projekcijų grafikai, kiekviena
# projekcija atskirame paveiksle (stilius toks pat kaip UMAP grafikų).
#   grafikai/parametrai/   – visi 8 deriniai (perplexity × init);
#   grafikai/sekios/       – atsitiktinė inicializacija su 5 sėklomis;
#   grafikai/jautrumas/    – B ir C imčių projekcijos;
#   grafikai/tsne_pasirinkta.png – galutinė projekcija ataskaitai.
#
# Projekcijos neskaičiuojamos iš naujo – naudojami .npy failai, kuriuos
# išsaugojo tsne_params.py, tsne_stability.py ir tsne_compare.py.

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "UMAP"))

from metrics import evaluate_projection
from umap_plots import metric_line, plot_projection

DATA_DIR = ROOT / "data"
TSNE_DIR = ROOT / "rezultatai" / "tsne"
PARAMS_DIR = TSNE_DIR / "projekcijos"
STABILITY_DIR = TSNE_DIR / "stabilumas"
COMPARE_DIR = TSNE_DIR / "jautrumas"
PARAMS_FILE = TSNE_DIR / "tsne_parametrai.csv"
STABILITY_FILE = TSNE_DIR / "tsne_stabilumas.csv"

FIGURES_DIR = TSNE_DIR / "grafikai"
PARAMS_FIGURES_DIR = FIGURES_DIR / "parametrai"
SEEDS_FIGURES_DIR = FIGURES_DIR / "sekios"
COMPARE_FIGURES_DIR = FIGURES_DIR / "jautrumas"

INIT_NAMES = {"pca": "PCA", "random": "atsitiktinė"}
CHOSEN = {"perplexity": 50, "init": "pca"}
SEEDS_PERPLEXITY = 50


def parameters_title(perplexity, init):
    return f"t-SNE: perplexity = {perplexity}, {INIT_NAMES[init]} inicializacija"


def plot_all_parameters(data, params):
    for row in params.to_dict("records"):
        name = f"tsne_p{row['perplexity']}_{row['init']}"
        Y = np.load(PARAMS_DIR / f"{name}.npy")
        title = parameters_title(row["perplexity"], row["init"]) + f"\n{metric_line(row)}"
        plot_projection(Y, data, title, PARAMS_FIGURES_DIR / f"{name}.png")
    print(f"  -> {PARAMS_FIGURES_DIR} ({len(params)} paveikslai)")


def plot_seeds(data, stability):
    rows = stability[(stability["perplexity"] == SEEDS_PERPLEXITY)
                     & (stability["init"] == "random")]
    for row in rows.to_dict("records"):
        name = f"tsne_p{SEEDS_PERPLEXITY}_random_seed{row['seed']}"
        Y = np.load(STABILITY_DIR / f"{name}.npy")
        title = (parameters_title(SEEDS_PERPLEXITY, "random") + f", sėkla {row['seed']}\n"
                 + metric_line(row))
        plot_projection(Y, data, title, SEEDS_FIGURES_DIR / f"{name}.png")
    print(f"  -> {SEEDS_FIGURES_DIR} ({len(rows)} paveikslai)")


def plot_chosen(data, params):
    row = params[(params["perplexity"] == CHOSEN["perplexity"])
                 & (params["init"] == CHOSEN["init"])].iloc[0]
    Y = np.load(PARAMS_DIR / f"tsne_p{CHOSEN['perplexity']}_{CHOSEN['init']}.npy")
    path = FIGURES_DIR / "tsne_pasirinkta.png"
    plot_projection(Y, data, parameters_title(**CHOSEN) + f"\n{metric_line(row)}", path)
    print("  ->", path)


def plot_compare():
    name = f"p{CHOSEN['perplexity']}_{CHOSEN['init']}"
    for subset in ["B", "C"]:
        path = COMPARE_DIR / f"tsne_{subset}_{name}.npy"
        if not path.exists():
            print(f"  {subset} projekcijos nėra – pirmiausia paleiskite tsne_compare.py")
            continue
        data = pd.read_csv(DATA_DIR / f"{subset}_norm.csv")
        features = [c for c in data.columns
                    if c not in ["id", "irasas", "R", "klase", "ritmas", "ekstremali_isskirtis"]]
        Y = np.load(path)
        metrics = evaluate_projection(data[features].to_numpy(), Y, data["klase"].to_numpy())
        title = f"{subset} aibė. " + parameters_title(**CHOSEN) + f"\n{metric_line(metrics)}"
        plot_projection(Y, data, title, COMPARE_FIGURES_DIR / f"tsne_{subset}_{name}.png")
    print(f"  -> {COMPARE_FIGURES_DIR}")


def main():
    for folder in [FIGURES_DIR, PARAMS_FIGURES_DIR, SEEDS_FIGURES_DIR, COMPARE_FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_DIR / "A_norm.csv")
    params = pd.read_csv(PARAMS_FILE)
    stability = pd.read_csv(STABILITY_FILE)

    print("t-SNE grafikai:")
    plot_chosen(data, params)
    plot_all_parameters(data, params)
    plot_seeds(data, stability)
    plot_compare()


if __name__ == "__main__":
    main()
