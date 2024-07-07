
import numpy as np
from scipy.stats import norm, multivariate_normal

# From https://github.com/rlabbe/filterpy/blob/master/filterpy/stats/stats.py#L131
def logpdf(x, mean=None, cov=1, allow_singular=True):
    """
    Computes the log of the probability density function of the normal
    N(mean, cov) for the data x. The normal may be univariate or multivariate.

    Wrapper for older versions of scipy.multivariate_normal.logpdf which
    don't support support the allow_singular keyword prior to verion 0.15.0.

    If it is not supported, and cov is singular or not PSD you may get
    an exception.

    `x` and `mean` may be column vectors, row vectors, or lists.
    """

    if mean is not None:
        flat_mean = np.asarray(mean).flatten()
    else:
        flat_mean = None

    flat_x = np.asarray(x).flatten()

    return multivariate_normal.logpdf(flat_x, flat_mean, cov, allow_singular)