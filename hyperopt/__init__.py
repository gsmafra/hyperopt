
from .base import STATUS_STRINGS
from .base import STATUS_NEW
from .base import STATUS_OK
from .base import STATUS_FAIL

from .base import JOB_STATES
from .base import JOB_STATE_NEW
from .base import JOB_STATE_RUNNING
from .base import JOB_STATE_DONE
from .base import JOB_STATE_ERROR

from .base import Ctrl
from .base import Trials
from .base import Domain

from .fmin import fmin
from .fmin import fmin_path
from .fmin import FMinIter

# -- syntactic sugar
import hp

# -- exceptions
import exceptions

# -- Import built-in optimization algorithms
import rand
import tpe
import mix
import anneal

__version__ = '0.0.3.gsmafra.0'
