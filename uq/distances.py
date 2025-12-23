
import numpy as np
from scipy.spatial import distance
from joblib import Parallel, delayed

class measure_distance:
    def __init__(self, observation, distribution):
        '''
        array1: test set array
        array2: train set array
        '''
        self.observation = observation
        self.distribution = distribution

    def Minkowski_distance(self, p=2, knn = 50,n_jobs=8, chunk_size=100):
        '''
        Compute the k-nearest Minkowski distances between rows of arr1[start:end] and arr2.
        p = order of norm
        n_jobs = number of cpu tasks for parallelization
        returns an array of shape (end-start, knn) containing the k smallest distances for each row
        '''
        n1, n2 = len(self.observation), len(self.distribution)
        print(f"Computing Minkowski distances: arr1={n1}, arr2={n2}, p={p}, jobs={n_jobs}")

        def compute_block(arr1, arr2, start, end, p, knn):
            """Compute distances between arr1[start:end] and arr2"""
            block = arr1[start:end]
            D = distance.cdist(block, arr2, metric='minkowski', p=p)
            print(f"Processed rows {start}-{end}", flush=True)
            D_smallest = np.partition(D, knn-1, axis=1)[:, :knn]
            return D_smallest

        blocks = [(i, min(i + chunk_size, n1)) for i in range(0, n1, chunk_size)]

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_block)(self.observation, self.distribution, start, end, p, knn) for start, end in blocks
        )

        distance_matrix = np.vstack(results)
        return distance_matrix
     

    def Mahalanobis_distance(self, alpha2 = 1):
        cov = np.cov(self.distribution, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        variance_matrix = []
        distribution_mean = np.mean(self.distribution, axis = 0)
        for obs in self.observation:
            cat_id = obs[-4]
            system_id = obs[-3]
            atom_id = obs[-2]
            test = obs[:-4]
            diff = test - distribution_mean
            m =  np.dot(np.dot(diff, inv_cov), diff.T)
            dist = alpha2 * np.sqrt(m)
            variance_matrix.append([dist, cat_id, system_id, atom_id])

        return variance_matrix
    
    def Convex_hull():
        pass

    

        