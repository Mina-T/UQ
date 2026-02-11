
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


class ArraySimilarityGPU:
    def __init__(self, observation_dict, distribution_dict, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Convert dicts to numpy arrays
        observation_array = np.array(list(observation_dict.values()))
        distribution_array = np.array(list(distribution_dict.values()))

        # Store keys for later
        self.observation_keys = list(observation_dict.keys())
        self.distribution_keys = list(distribution_dict.keys())

        # Convert to tensors
        self.obs_tensor = torch.tensor(observation_array, dtype=torch.float32, device=self.device)
        self.dist_tensor = torch.tensor(distribution_array, dtype=torch.float32, device=self.device)

        # Normalize to probability distributions for KL divergence
        self._normalize_tensors()

    # assume that each atomic representation is a distribution
    def _normalize_tensors(self, eps=1e-12):
        """Normalize tensors to valid probability distributions"""
        # Make sure all values are positive (shift if negative)
        obs_min = self.obs_tensor.min(dim=1, keepdim=True)[0]
        dist_min = self.dist_tensor.min(dim=1, keepdim=True)[0]

        if (obs_min < 0).any():
            self.obs_tensor = self.obs_tensor - obs_min + eps
        if (dist_min < 0).any():
            self.dist_tensor = self.dist_tensor - dist_min + eps

        # Normalize to sum to 1 (probability distribution)
        self.obs_tensor = self.obs_tensor / (self.obs_tensor.sum(dim=1, keepdim=True) + eps)
        self.dist_tensor = self.dist_tensor / (self.dist_tensor.sum(dim=1, keepdim=True) + eps)

    def KL_divergence(self, batch_size_obs=64, batch_size_dist=2000, k_nearest=None, eps=1e-12):
        """
        Compute KL(observation ‖ distribution) for each observation vector
        
        Returns: dict {observation_key: mean KL to k-nearest distributions}
        """
        n_obs = self.obs_tensor.shape[0]
        n_dist = self.dist_tensor.shape[1]

        print(f"Observations: {n_obs} vectors, Dimensions: {n_dist}", flush=True)
        print(f"Distributions: {self.dist_tensor.shape[0]} vectors", flush=True)

        kl_results = {}

        for obs_start in range(0, n_obs, batch_size_obs):
            obs_end = min(obs_start + batch_size_obs, n_obs)
            obs_batch = self.obs_tensor[obs_start:obs_end]  # (batch_size, n_dim)
            obs_keys = self.observation_keys[obs_start:obs_end]

            batch_kl_all = []
            for dist_start in range(0, self.dist_tensor.shape[0], batch_size_dist):
                dist_end = min(dist_start + batch_size_dist, self.dist_tensor.shape[0])
                dist_batch = self.dist_tensor[dist_start:dist_end]  # (chunk_size, n_dim)

                dist_exp = dist_batch.unsqueeze(1)  # Distributions: p
                obs_exp = obs_batch.unsqueeze(0)    # Observations: q

                kl_batch = torch.sum(
                    dist_exp * torch.log((dist_exp + eps) / (obs_exp + eps)),
                    dim=2
                )  

                batch_kl_all.append(kl_batch)

                del dist_exp, obs_exp, kl_batch
                torch.cuda.empty_cache()
            
            batch_kl_all = torch.cat(batch_kl_all, dim=0)

            if k_nearest:

                k_to_use = min(k_nearest, batch_kl_all.shape[0])


                smallest_kl, _ = torch.topk(batch_kl_all, k=k_to_use, dim=0, largest=False)
                mean_kl_values = smallest_kl.mean(dim=0)  # Average over k-nearest

                for key, kl_value in zip(obs_keys, mean_kl_values):
                    kl_results[key] = kl_value.item()

            del batch_kl_all, obs_batch
            torch.cuda.empty_cache()

            print(f"Processed observations {obs_end}/{n_obs}", flush=True)

        print(f"✅ Done! Computed KL divergence for {len(kl_results)} observations.")
        return kl_results

                





    

        
