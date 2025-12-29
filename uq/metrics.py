import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

def nllf(model_err, model_std_dev, eps=1e-12):
    model_err = np.linalg.norm(np.asarray(model_err))
    model_std_dev = np.linalg.norm(np.asarray(model_std_dev))
    var = model_std_dev**2
    normalized_err = (model_err)**2 / var
    sm = (np.log(2* np.pi*var)  + normalized_err).mean()
    nll = 0.5*sm
    return nll

def R2(y_ref, y_pred):
    return r2_score(y_ref, y_pred)


def pearson(x, y):
    pearson_corr, p = pearsonr(x, y)
    return pearson_corr


def spearsman(x, y):
    spearman_corr, p = spearmanr(x, y)
    return spearman_corr
