# 8 uždavinys: projekcijų kokybės vertinimo funkcijos.
# Šį failą importuoja visi metodų skriptai (PCA, MDS, t-SNE, UMAP):
#
#     from metrics import evaluate_projection
#     result = evaluate_projection(X, Y, labels)
#
# X - normuoti pradiniai požymiai (be klasės!), Y - projekcija,
# labels – klasės, naudojamos TIK NH ir kNN tikslumui.
#
# Paleidus failą tiesiogiai (python metrics.py), funkcijos patikrinamos
# su Iris duomenimis ir palyginamos su skaidrėse pateiktomis reikšmėmis.

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

K_VALUES = (5, 15, 50)      # kaimynystės dydžiai pagal skaidres
NH_K = 10                   # kaimynų skaičius Neighborhood Hit matui
KNN_CLASSIFIER_K = 5        # kNN klasifikatoriaus kaimynų skaičius
CV_FOLDS = 10               # kryžminės patikros blokų skaičius
RANDOM_STATE = 42


# Lokali struktūra
def trustworthiness(X, Y, k):
    # Ar projekcijos kaimynai yra tikri kaimynai pradinėje erdvėje?
    # Mažėja, kai projekcijoje atsiranda "netikrų" kaimynų.
    return sklearn_trustworthiness(X, Y, n_neighbors=k)


def continuity(X, Y, k):
    # Ar pradinės erdvės kaimynai liko kaimynais projekcijoje?
    # Tai ta pati formulė kaip trustworthiness, tik sukeistos erdvės.
    return sklearn_trustworthiness(Y, X, n_neighbors=k)


def nearest_neighbors(data, k):
    # k artimiausių kaimynų indeksai (pats taškas neįtraukiamas).
    model = NearestNeighbors(n_neighbors=k + 1).fit(data)
    indices = model.kneighbors(data, return_distance=False)
    return indices[:, 1:]


def neighbor_overlap(first, second, k):
    # Vidutinė dalis tų pačių k kaimynų dviejose erdvėse (0–1).
    # Naudojama ir kNN išsaugojimui (X ir Y), ir dviejų projekcijų palyginimui.
    first_neighbors = nearest_neighbors(first, k)
    second_neighbors = nearest_neighbors(second, k)
    shared = [
        len(set(a) & set(b)) for a, b in zip(first_neighbors, second_neighbors)
    ]
    return np.mean(shared) / k


# Globali struktūra
def distance_correlation(X, Y):
    # Spearman koreliacija tarp porinių atstumų pradinėje erdvėje ir projekcijoje.
    rho, _ = spearmanr(pdist(X), pdist(Y))
    return rho


def stress_1(X, Y):
    # Normuotas Stress-1 su optimaliu mastelio koeficientu:
    # sqrt( sum (d_X - a * d_Y)^2 / sum d_X^2 ), a = sum(d_X * d_Y) / sum(d_Y^2).
    # Mastelio koeficientas reikalingas, nes t-SNE ir UMAP ašių vienetai
    # neturi ryšio su pradiniais atstumais.
    distances_x = pdist(X)
    distances_y = pdist(Y)
    scale = (distances_x * distances_y).sum() / (distances_y ** 2).sum()
    return np.sqrt(((distances_x - scale * distances_y) ** 2).sum() / (distances_x ** 2).sum())


# Klasių atskyrimas (čia, ir tik čia, naudojamos klasės)
def neighborhood_hit(Y, labels, k=NH_K):
    # Vidutinė dalis projekcijos kaimynų, priklausančių tai pačiai klasei.
    labels = np.asarray(labels)
    neighbors = nearest_neighbors(Y, k)
    return (labels[neighbors] == labels[:, None]).mean()


def knn_accuracy(Y, labels, k=KNN_CLASSIFIER_K):
    # kNN klasifikatoriaus tikslumas projekcijoje (10 blokų kryžminė patikra).
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(KNeighborsClassifier(k), Y, labels, cv=folds)
    return scores.mean()


# Viena funkcija visiems metodams
def evaluate_projection(X, Y, labels=None, k_values=K_VALUES):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    result = {}
    for k in k_values:
        result[f"T_{k}"] = trustworthiness(X, Y, k)
        result[f"C_{k}"] = continuity(X, Y, k)
        result[f"kNN_{k}"] = neighbor_overlap(X, Y, k)

    result["Spearman_rho"] = distance_correlation(X, Y)
    result["Stress_1"] = stress_1(X, Y)

    if labels is not None:
        result[f"NH_{NH_K}"] = neighborhood_hit(Y, labels)
        result["kNN_tikslumas"] = knn_accuracy(Y, labels)

    return {name: float(value) for name, value in result.items()}


# Patikra su Iris duomenimis
def check_with_iris():
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Skaidrėse naudojami standartizuoti Iris duomenys (PC1 + PC2 = 95,8 %).
    X, labels = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)

    result = evaluate_projection(X, Y, labels, k_values=(10,))

    expected = {
        "PC1+PC2 dispersija": (0.958, pca.explained_variance_ratio_.sum()),
        "T_10": (0.978, result["T_10"]),
        "C_10": (0.991, result["C_10"]),
        "Spearman_rho": (0.993, result["Spearman_rho"]),
        "Stress_1": (0.060, result["Stress_1"]),
        "kNN_tikslumas": (0.907, result["kNN_tikslumas"]),
    }

    print("8 uždavinys: metrikų patikra su Iris (PCA, 2D)\n")
    print(f"{'Metrika':<20}{'Skaidrėse':>10}{'Gauta':>10}{'Sutampa':>10}")
    all_match = True
    for name, (slide_value, our_value) in expected.items():
        match = abs(round(our_value, 3) - slide_value) <= 0.001
        all_match &= match
        print(f"{name:<20}{slide_value:>10.3f}{our_value:>10.3f}{'taip' if match else 'NE':>10}")

    print("\nVisos metrikos sutampa su skaidrėmis." if all_match
          else "\nDĖMESIO: ne visos metrikos sutampa su skaidrėmis.")


if __name__ == "__main__":
    check_with_iris()