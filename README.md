## Dimensionality Reduction Algorithms

The project studies how four dimensionality reduction methods (PCA, MDS, t-SNE and UMAP) preserve the structure of heartbeat data when it is projected into 2D.

### Data 
- **Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) (PhysioNet), prepared in the first assignment ([repository](https://github.com/AnastasijaFed/duomenu-aibes-apdorojimas)).
- **Classes:** 1000 beats from each of 5 classes – N (normal), L (left bundle branch block), R (right bundle branch block), V (premature ventricular contraction), A (atrial premature beat).

### Samples

| Sample | Description | Objects | Features |
|---|---|---|---|
| **A** | main sample | 4837 | 5 |
| **B** | A without the strongly correlated RR_sant_post | 4837 | 4 |
| **C** | A without the 366 extreme outliers | 4471 | 5 |

All samples are normalised with robust scaling: (x − median) / IQR.

### Methods

- **PCA** – implemented from scratch (covariance matrix, eigenvectors), verified against `scikit-learn`.
- **MDS** – SMACOF algorithm, metric and non-metric variants, initialised from PCA.
- **t-SNE** – perplexity (5–50) and initialisation (PCA / random) studied.
- **UMAP** – `n_neighbors`, `min_dist` and initialisation studied, stability tested with 5 seeds.

### Quality metrics

All methods are evaluated with the same function (`metrics.py`):

- **local structure:** trustworthiness, continuity, kNN neighbour preservation (k = 5, 15, 50);
- **global structure:** Spearman ρ between pairwise distances, Stress-1;
- **class separation:** Neighborhood Hit, kNN accuracy.

### Conclusions

t-SNE and UMAP best preserve local neighbourhoods and class separation, while PCA and MDS best preserve pairwise distances, so the methods complement rather than replace each other. In all projections, classes V and A separate from the rest while N, L and R overlap, and removing the correlated feature or the extreme outliers did not reveal a better structure, so the full sample A with marked outliers is the most suitable. Overall, t-SNE is best for visualisation, metric MDS for judging distances between objects, and UMAP for projecting new objects.
