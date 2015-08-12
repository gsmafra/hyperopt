"""
Extra distributions to complement scipy.stats

"""
import numpy as np
import numpy.random as mtrand
import scipy.stats
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats.distributions import rv_generic


class uniform_gen(scipy.stats.distributions.uniform_gen):
    # -- included for completeness
    pass


class norm_gen(scipy.stats.distributions.norm_gen):
    # -- included for completeness
    pass


def qtable_pmf(x, q, qlow, xs, ps):
    qx = np.round(np.atleast_1d(x).astype(np.float) / q) * q
    is_multiple = np.isclose(qx, x)
    ix = np.round((qx - qlow) / q).astype(np.int)
    is_inbounds = np.logical_and(ix >= 0, ix < len(ps))
    oks = np.logical_and(is_multiple, is_inbounds)
    rval = np.zeros_like(qx)
    rval[oks] = np.asarray(ps)[ix[oks]]
    if isinstance(x, np.ndarray):
        return rval.reshape(x.shape)
    else:
        return float(rval)


def qtable_logpmf(x, q, qlow, xs, ps):
    p = qtable_pmf(np.atleast_1d(x), q, qlow, xs, ps)
    # -- this if/else avoids np warning about underflow
    rval = np.zeros_like(p)
    rval[p == 0] = -np.inf
    rval[p != 0] = np.log(p[p != 0])
    if isinstance(x, np.ndarray):
        return rval
    else:
        return float(rval)

