import numpy as np
import pickle
from utils.labels import drautz_atom_label
from dadapy import (
    DensityAdvanced,
    Clustering,
    Data
)

class compute_density:
    """
    Computes density-based scores for observations wrt a reference distribution
    and returns a dictionary mapping labels to interpolated densities.
    """
    def __init__(self, observation, distribution, model_name,maxk= 50):
        self.observation = observation
        self.distribution = distribution
        self.maxk = maxk
        self.model_name = model_name

    # find your own version of this func
    @staticmethod
    def return_unique_point_indices(distances, dist_indices, tol=1.01 * np.finfo(np.float32).eps):
        """
        Find unique representatives of identical points within a tolerance tol.
        Return indices of one unique representative per duplicate group (original order).
        Used to preprocess data to avoid issues with identical points in ID and density estimation.
        distances: (N,k+1) array of distances to k nearest neighbors (including self at 0 distance).
        dist_indices: (N,k+1) array of indices of the k nearest neighbors (including self at index 0).
        tol: tolerance within which points are considered identical.
        Use as:
        unique_indices = return_unique_point_indices(distances, dist_indices, tol)
        data_unique = data[unique_indices]
        """
        N, k = distances.shape
        parent = np.arange(N, dtype=int)
        rank = np.zeros(N, dtype=int)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x


        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Use ALL columns; add edges for every neighbor within tol (excluding true self)
        mask = np.isfinite(distances) & (distances <= tol)
        rows, cols = np.where(mask)
        for i, j in zip(rows, cols):
            nb = int(dist_indices[i, j])
            if nb != i:          # ignore true self if present
                union(i, nb)

        # pick the first (lowest) original index per component; return in original order
        rep_for_root = {}
        for i in range(N):
            r = find(i)
            rep_for_root[r] = min(rep_for_root.get(r, i), i)

        return np.array(sorted(rep_for_root.values()), dtype=int)

    def get_density(self):
        labels = np.array([
        drautz_atom_label(cat, sys_id, atom_id)
        for cat, sys_id, atom_id in self.observation[:, -3:]
        ])
        _all = np.concatenate((self.distribution , self.observation), axis = 0)
        X = _all[:, :-3]
        origin = np.concatenate((
            np.zeros(len(self.distribution), dtype=int),
            np.ones(len(self.observation), dtype=int)
        ))

        de = Data(coordinates=X, maxk=self.maxk, verbose=True)
        de.compute_distances()
        unique_idx = self.return_unique_point_indices(de.distances, de.dist_indices)
        X_unique = X[unique_idx]
        origin_unique = origin[unique_idx]
        mask_test_unique = (origin_unique == 1)
        X_unique_test = X_unique[mask_test_unique]
        X_unique_labels = labels[mask_test_unique]
        print(len(X) - len(X_unique), ' points removed.', flush = True)
        du = Data(coordinates=X_unique[::1], maxk=self.maxk, verbose=True)
        du.compute_distances()
        du.compute_id_2NN()
        du.compute_density_kstarNN()
        interpolated_logden, _ = du.return_interpolated_density_kstarNN(X_unique_test)
        density_dict = {label: density for label, density in zip(X_unique_labels, interpolated_logden)}
        with open(f'test_density_dict_{self.model_name}.pkl', 'wb') as f:
            pickle.dump(density_dict, f)                                                                



