# Pasirinktos UMAP projekcijos interpretacijos patikra.
# Projekcijoje automatiškai randamos grupės (HDBSCAN) ir kiekvienai
# parodoma: kiek joje taškų, kokia dalis yra ekstremalios išskirtys,
# kokios klasės, įrašai ir ritmai, RR_pre mediana.
# Taip pat patikrinama, kiek N klasės išskirčių atsiduria V ir A srityje.
#
# Įvestis: data/A.csv (tikros požymių reikšmės),
#          rezultatai/umap/projekcijos/umap_nn50_md0.1_spectral.npy

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = Path("data") / "A.csv"
PROJECTION_FILE = Path("rezultatai") / "umap" / "projekcijos" / "umap_nn50_md0.1_spectral.npy"

MIN_CLUSTER_SIZE = 40
OUTLIER_GROUP_SHARE = 0.5   # grupė laikoma „išskirčių grupe“, jei jų > 50 %


def describe(group):
    return {
        "dūžių": len(group),
        "išskirčių_%": round(group["ekstremali_isskirtis"].mean() * 100, 1),
        "klasės": group["klase"].value_counts().to_dict(),
        "įrašai (3 dažniausi)": group["irasas"].value_counts().head(3).to_dict(),
        "ritmai (3 dažniausi)": group["ritmas"].value_counts().head(3).to_dict(),
        "RR_pre mediana, ms": round(group["RR_pre"].median()),
    }


def main():
    data = pd.read_csv(DATA_FILE)
    Y = np.load(PROJECTION_FILE)
    data["grupe"] = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(Y)

    print(f"Visa A imtis: RR_pre mediana {data['RR_pre'].median():.0f} ms, "
          f"išskirčių {data['ekstremali_isskirtis'].mean() * 100:.1f} %\n")

    print("Grupės, kuriose daugiau nei pusė taškų yra ekstremalios išskirtys:")
    found = False
    for label, group in data[data["grupe"] >= 0].groupby("grupe"):
        if group["ekstremali_isskirtis"].mean() > OUTLIER_GROUP_SHARE:
            found = True
            for name, value in describe(group).items():
                print(f"  {name}: {value}")
            print()
    if not found:
        print("  nerasta\n")

    # V ir A sritis – grupės, kuriose daugiau nei pusė dūžių yra V arba A.
    va_share = data.groupby("grupe")["klase"].apply(lambda c: c.isin(["V", "A"]).mean())
    va_groups = [g for g, share in va_share.items() if g >= 0 and share > 0.5]
    in_va = data["grupe"].isin(va_groups)

    n_outliers = data[(data["klase"] == "N") & data["ekstremali_isskirtis"]]
    n_in_va = n_outliers[in_va.loc[n_outliers.index]]
    print(f"N klasės išskirčių iš viso: {len(n_outliers)}, iš jų V ir A srityje: {len(n_in_va)}")
    print(f"  jų ritmai: {n_in_va['ritmas'].value_counts().head(3).to_dict()}")


if __name__ == "__main__":
    main()