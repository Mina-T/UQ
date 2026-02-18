import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

class measure_correlation:
    def __init__(self, uq_metric, ens_uncertainty):
        self.uq_metric = uq_metric
        self.ens_uncertainty = ens_uncertainty

    def nllf(self, eps=1e-12):
        uq_metric = np.linalg.norm(np.asarray(self.uq_metric))
        ens_uncertainty = np.linalg.norm(np.asarray(self.ens_uncertainty))
        var = ens_uncertainty**2
        normalized_err = (uq_metric)**2 / var
        sm = (np.log(2* np.pi*var)  + normalized_err).mean()
        nll = 0.5*sm
        return nll

    def R2(self):
        return r2_score(self.uq_metric, self.ens_uncertainty)


    def pearson(self):
        pearson_corr, p = pearsonr(self.uq_metric, self.ens_uncertainty)
        return pearson_corr


    def spearsman(self):
        spearman_corr, p = spearmanr(self.uq_metric, self.ens_uncertainty)
    return spearman_corr
