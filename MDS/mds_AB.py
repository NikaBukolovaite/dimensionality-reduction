"ka darom: MDS A ir B aibems (B = A be RR_sant_post), abiem variantais (metrinis ir nemetrinis). Lyginam ar pasalinus koreliuota pozymi isliko ta pati objektu struktura. Objektai A ir B tie patys, todel lyginam tiesiogiai."

# mds importuojamas pirmas: jis nustato kelią iki metrics.py, PCA/ ir data/
from mds import run_mds, plot_projection, VARIANTS, FEATURES

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from metrics import evaluate_projection, neighbor_overlap, K_VALUES


FILE_A = Path("data/A_norm.csv")
FILE_B = Path("data/B_norm.csv")
OUTPUT_DIR = Path("rezultatai/mds/AB")

FEATURES_A = FEATURES
FEATURES_B = [f for f in FEATURES if f != "RR_sant_post"]


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

    # pradinių erdvių palyginimas nepriklauso nuo MDS varianto
    original = {"rho(D_A, D_B) pradinės erdvės": spearmanr(pdist(X_a), pdist(X_b))[0]}
    for k in K_VALUES:
        original[f"kaimynų sutapimas X, k={k}"] = neighbor_overlap(X_a, X_b, k)
    original = pd.Series(original, name="reikšmė")
    print("Pradinių erdvių palyginimas:")
    print(original.round(3), "\n")

    metrics = {}
    comparison = {}
    for variant in VARIANTS:
        print(f"(skaiciuoja {variant} MDS A ir B, ~2 min)")
        Y_a = run_mds(X_a, variant)
        Y_b = run_mds(X_b, variant)

        metrics[f"A {variant}"] = evaluate_projection(X_a, Y_a, labels)
        metrics[f"B {variant}"] = evaluate_projection(X_b, Y_b, labels)

        comparison[variant] = {"rho(Y_A, Y_B) projekcijos": spearmanr(pdist(Y_a), pdist(Y_b))[0]}
        for k in K_VALUES:
            comparison[variant][f"kaimynų sutapimas Y, k={k}"] = neighbor_overlap(Y_a, Y_b, k)

        np.save(OUTPUT_DIR / f"mds_B_{variant}.npy", Y_b)
        plot_projection(Y_b, data_b, f"MDS ({variant}): B aibės projekcija (be RR_sant_post)",
                        OUTPUT_DIR / f"mds_B_{variant}.png")

    metrics = pd.DataFrame(metrics)
    for variant in VARIANTS:
        metrics[f"pokytis {variant}"] = metrics[f"B {variant}"] - metrics[f"A {variant}"]
    comparison = pd.DataFrame(comparison)

    print("\nProjekcijos kokybė, A ir B:")
    print(metrics.round(3), "\n")
    print("A ir B projekcijų palyginimas:")
    print(comparison.round(3), "\n")

    metrics.round(4).to_csv(OUTPUT_DIR / "mds_AB_metrikos.csv", encoding="utf-8-sig")
    comparison.round(4).to_csv(OUTPUT_DIR / "mds_AB_palyginimas.csv", encoding="utf-8-sig")
    original.round(4).to_csv(OUTPUT_DIR / "mds_AB_pradines_erdves.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()