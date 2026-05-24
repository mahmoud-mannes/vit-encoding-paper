import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist


def spatial_similarity_distance_correlation(S,grid_size, metric):

    T = S.shape[0]
    assert S.shape[0] == S.shape[1], "Similarity matrix must be square (T×T)"
    assert grid_size * grid_size == T, "grid_size^2 must equal T"

    # Spatial coordinates (row-major)
    coords = np.stack(
        np.meshgrid(
            np.arange(grid_size),
            np.arange(grid_size),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)  # (T, 2)

    # Pairwise spatial distances
    if metric == "manhattan":
        dists = cdist(coords, coords, metric="cityblock")
    elif metric == "euclidean":
        dists = cdist(coords, coords, metric="euclidean")
    else:
        raise ValueError("metric must be 'manhattan' or 'euclidean'")

    # Upper triangle (exclude diagonal)
    iu = np.triu_indices(T, k=1)

    sim_vals = S[iu]
    dist_vals = dists[iu]

    corr, _ = spearmanr(-dist_vals, sim_vals)
    return corr