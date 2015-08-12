try:
    import dill as cPickle
except ImportError:
    import cPickle

import functools
import logging
import os
import sys
import time

import numpy as np

from .utils import coarse_utcnow
from . import base

logger = logging.getLogger(__name__)


class FMinIter(object):
    """Object for conducting search experiments.
    """
    catch_eval_exceptions = False
    cPickle_protocol = -1

    def __init__(self, algo, domain, trials, rstate, 
            max_queue_len=1,
            poll_interval_secs=1.0,
            max_evals=sys.maxint,
            ):
        self.algo = algo
        self.domain = domain
        self.trials = trials
        self.poll_interval_secs = poll_interval_secs
        self.max_queue_len = max_queue_len
        self.max_evals = max_evals
        self.rstate = rstate


    def serial_evaluate(self, N=-1):
        for trial in self.trials._dynamic_trials:
            if trial['state'] == base.JOB_STATE_NEW:
                trial['state'] == base.JOB_STATE_RUNNING
                now = coarse_utcnow()
                trial['book_time'] = now
                trial['refresh_time'] = now
                spec = base.spec_from_misc(trial['misc'])
                ctrl = base.Ctrl(self.trials, current_trial=trial)
                try:
                    result = self.domain.evaluate(spec, ctrl)
                except Exception, e:
                    logger.info('job exception: %s' % str(e))
                    trial['state'] = base.JOB_STATE_ERROR
                    trial['misc']['error'] = (str(type(e)), str(e))
                    trial['refresh_time'] = coarse_utcnow()
                    if not self.catch_eval_exceptions:
                        self.trials.refresh()
                        raise
                else:
                    trial['state'] = base.JOB_STATE_DONE
                    trial['result'] = result
                    trial['refresh_time'] = coarse_utcnow()
                N -= 1
                if N == 0:
                    break
        self.trials.refresh()

    def run(self, N):
    
        trials = self.trials
        algo = self.algo
        n_queued = 0

        def get_queue_len():
            return self.trials.count_by_state_unsynced(base.JOB_STATE_NEW)

        stopped = False
        while n_queued < N:
            qlen = get_queue_len()
            while qlen < self.max_queue_len and n_queued < N:
                n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                new_ids = trials.new_trial_ids(n_to_enqueue)
                self.trials.refresh()
                if 0:
                    for d in self.trials.trials:
                        print 'trial %i %s %s' % (d['tid'], d['state'],
                            d['result'].get('status'))
                new_trials = algo(new_ids, self.domain, trials,
                                  self.rstate.randint(2 ** 31 - 1))
                assert len(new_ids) >= len(new_trials)
                if len(new_trials):
                    self.trials.insert_trial_docs(new_trials)
                    self.trials.refresh()
                    n_queued += len(new_trials)
                    qlen = get_queue_len()
                else:
                    stopped = True
                    break

            # -- loop over trials and do the jobs directly
            self.serial_evaluate()

            if stopped:
                break

        qlen = get_queue_len()
        if qlen:
            msg = 'Exiting run, not waiting for %d jobs.' % qlen
            logger.info(msg)

    def __iter__(self):
        return self

    def exhaust(self):
        n_done = len(self.trials)
        self.run(self.max_evals - n_done)
        self.trials.refresh()
        return self


def fmin(fn, space, algo, max_evals, trials=None, rstate=None,
         pass_expr_memo_ctrl=None,
         catch_eval_exceptions=False,
         return_argmin=True,
        ):
        
    if rstate is None:
        env_rseed = os.environ.get('HYPEROPT_FMIN_SEED', '')
        if env_rseed:
            rstate = np.random.RandomState(int(env_rseed))
        else:
            rstate = np.random.RandomState()

    domain = base.Domain(fn, space,
        pass_expr_memo_ctrl=pass_expr_memo_ctrl)

    rval = FMinIter(algo, domain, trials, max_evals=max_evals,
                    rstate=rstate,)
    rval.catch_eval_exceptions = catch_eval_exceptions
    rval.exhaust()


### Fork
import tpe


def dump_trials(trials, path):

    print('Size of object: ' + str(len(trials)))
    print('Dumping')
    file = open(path, 'wb')
    cPickle.dump(trials, file)
    file.close()


def fmin_path(objective, space, max_evals, path):

    try:
        print 'Trying to pickle file'
        file = open(path, 'rb')
        trials = cPickle.load(file)
        file.close()
    except:
        print 'No trial file at specified path, creating new one'
        trials = base.Trials()
    else:
        print 'File found'

    try:
        print('Size of object: ' + str(len(trials)))
        fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + max_evals, trials=trials)
        dump_trials(trials, path)
    except KeyboardInterrupt:
        dump_trials(trials, path)
        raise

# -- flake8 doesn't like blank last line
