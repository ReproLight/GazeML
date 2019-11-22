"""Default specification of a data source."""
from collections import OrderedDict
import multiprocessing
import queue
import threading
import time

import numpy as np
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class BaseDataSource(object):
    """Base DataSource class."""

    pass

