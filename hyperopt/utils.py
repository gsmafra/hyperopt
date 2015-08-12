import datetime
import numpy as np
import logging
import cPickle
import os
import shutil
logger = logging.getLogger(__name__)

import numpy

import pyll

def import_tokens(tokens):
    # XXX Document me
    # import as many as we can
    rval = None
    for i in range(len(tokens)):
        modname = '.'.join(tokens[:i+1])
        # XXX: try using getattr, and then merge with load_tokens
        try:
            logger.info('importing %s' % modname)
            exec "import %s" % modname
            exec "rval = %s" % modname
        except ImportError, e:
            logger.info('failed to import %s' % modname)
            logger.info('reason: %s' % str(e))
            break
    return rval, tokens[i:]

def load_tokens(tokens):
    # XXX: merge with import_tokens
    logger.info('load_tokens: %s' % str(tokens))
    symbol, remainder = import_tokens(tokens)
    for attr in remainder:
        symbol = getattr(symbol, attr)
    return symbol


def json_lookup(json):
    symbol = load_tokens(json.split('.'))
    return symbol


def json_call(json, args=(), kwargs=None):
    """
    Return a dataset class instance based on a string, tuple or dictionary

    .. code-block:: python

        iris = json_call('datasets.toy.Iris')

    This function works by parsing the string, and calling import and getattr a
    lot. (XXX)

    """
    if kwargs is None:
        kwargs = {}
    if isinstance(json, basestring):
        symbol = json_lookup(json)
        return symbol(*args, **kwargs)
    elif isinstance(json, dict):
        raise NotImplementedError('dict calling convention undefined', json)
    elif isinstance(json, (tuple, list)):
        raise NotImplementedError('seq calling convention undefined', json)
    else:
        raise TypeError(json)


def get_obj(f, argfile=None, argstr=None, args=(), kwargs=None):
    """
    XXX: document me
    """
    if kwargs is None:
        kwargs = {}
    if argfile is not None:
        argstr = open(argfile).read()
    if argstr is not None:
        argd = cPickle.loads(argstr)
    else:
        argd = {}
    args = args + argd.get('args',())
    kwargs.update(argd.get('kwargs',{}))
    return json_call(f, args=args, kwargs=kwargs)


def pmin_sampled(mean, var, n_samples=1000, rng=None):
    """Probability that each Gaussian-dist R.V. is less than the others

    :param vscores: mean vector
    :param var: variance vector

    This function works by sampling n_samples from every (gaussian) mean distribution,
    and counting up the number of times each element's sample is the best.

    """
    if rng is None:
        rng = numpy.random.RandomState(232342)

    samples = rng.randn(n_samples, len(mean)) * numpy.sqrt(var) + mean
    winners = (samples.T == samples.min(axis=1)).T
    wincounts = winners.sum(axis=0)
    assert wincounts.shape == mean.shape
    return wincounts.astype('float64') / wincounts.sum()


def use_obj_for_literal_in_memo(expr, obj, lit, memo):
    """
    Set `memo[node] = obj` for all nodes in expr such that `node.obj == lit`

    This is a useful routine for fmin-compatible functions that are searching
    domains that include some leaf nodes that are complicated
    runtime-generated objects. One option is to make such leaf nodes pyll
    functions, but it can be easier to construct those objects the normal
    Python way in the fmin function, and just stick them into the evaluation
    memo.  The experiment ctrl object itself is inserted using this technique.
    """
    for node in pyll.dfs(expr):
        try:
            if node.obj == lit:
                memo[node] = obj
        except AttributeError:
            # -- non-literal nodes don't have node.obj
            pass
    return memo


def coarse_utcnow():
    """
    # MongoDB stores only to the nearest millisecond
    # This is mentioned in a footnote here:
    # http://api.mongodb.org/python/current/api/bson/son.html#dt
    """
    now = datetime.datetime.utcnow()
    microsec = (now.microsecond // 10 ** 3) * (10 ** 3)
    return datetime.datetime(now.year, now.month, now.day, now.hour,
                             now.minute, now.second, microsec)


