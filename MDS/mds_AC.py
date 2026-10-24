"ka darom: MDS A ir C aibems (C = A be ekstremaliu isskirciu), abiem variantais (metrinis ir nemetrinis). Lyginam ar isskirtys keite likusiu objektu struktura. Objektu skaicius skiriasi, todel A projekcijoje paliekam tik bendrus objektus (pagal id)."

from mds import run_mds, plot_projection, VARIANTS, FEATURES

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from metrics import evaluate_projection, neighbor_overlap, K_VALUES


FILE_A = Path("data/A_norm.csv")
FILE_C = Path("data/C_norm.csv")
OUTPUT_DIR = Path("rezultatai/mds/AC")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_a = pd.read_csv(FILE_A)
    data_c = pd.read_csv(FILE_C)

    X_a = data_a[FEATURES].to_numpy()
    X_c = data_c[FEATURES].to_numpy()
    labels_c = data_c["klase"].to_numpy()
    print(f"A: {X_a.shape[0]} objektų")
    print(f"C: {X_c.shape[0]} objektų (pašalinta {X_a.shape[0] - X_c.shape[0]} išskirčių)\n")

    # bendri objektai (tie, kurie yra ir C)
    common = data_a["id"].isin(data_c["id"]).to_numpy()
    assert (data_a.loc[common, "id"].to_numpy() == data_c["id"].to_numpy()).all(), \
        "bendrų objektų tvarka A ir C turi sutapti"
    X_a_common = X_a[common]

    metrics = {}
    comparison = {}
    for variant in VARIANTS:
        print(f"(skaiciuoja {variant} MDS A ir C, ~2 min)")

        # 1. MDS su visais objektais (A), vertinimui paliekam tik bendrus objektus
        Y_a_common = run_mds(X_a, variant)[common]

        # 2. MDS iš naujo be išskirčių (C)
        Y_c = run_mds(X_c, variant)

        # 3. Projekcijų kokybė tik bendriems objektams
        metrics[f"A {variant} (bendri)"] = evaluate_projection(X_a_common, Y_a_common, labels_c)
        metrics[f"C {variant}"] = evaluate_projection(X_c, Y_c, labels_c)

        # 4. Bendrų objektų atstumai ir kaimynystės A ir C projekcijose
        comparison[variant] = {"rho(Y_A, Y_C) bendri objektai": spearmanr(pdist(Y_a_common), pdist(Y_c))[0]}
        for k in K_VALUES:
            comparison[variant][f"kaimynų sutapimas Y, k={k}"] = neighbor_overlap(Y_a_common, Y_c, k)

        np.save(OUTPUT_DIR / f"mds_C_{variant}.npy", Y_c)
        plot_projection(Y_c, data_c, f"MDS ({variant}): C aibės projekcija (be išskirčių)",
                        OUTPUT_DIR / f"mds_C_{variant}.png")

    metrics = pd.DataFrame(metrics)
    for variant in VARIANTS:
        metrics[f"pokytis {variant}"] = metrics[f"C {variant}"] - metrics[f"A {variant} (bendri)"]
    comparison = pd.DataFrame(comparison)

    print("\nProjekcijos kokybė, A ir C (bendri objektai):")
    print(metrics.round(3), "\n")
    print("A ir C projekcijų palyginimas:")
    print(comparison.round(3), "\n")

    metrics.round(4).to_csv(OUTPUT_DIR / "mds_AC_metrikos.csv", encoding="utf-8-sig")
    comparison.round(4).to_csv(OUTPUT_DIR / "mds_AC_palyginimas.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()