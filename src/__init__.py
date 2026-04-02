"""
EEG2Img Benchmark Study
~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive benchmark study comparing time-series-to-image transformations
for EEG classification across diverse BCI paradigms.

:copyright: (c) 2026 by Research Team.
:license: MIT, see LICENSE for more details.
"""

__version__ = '0.1.0'
__author__ = 'Research Team'
__license__ = 'MIT'

from . import data
from . import transforms
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    'data',
    'transforms',
    'models',
    'training',
    'evaluation',
    'utils',
]
