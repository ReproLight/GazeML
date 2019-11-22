"""Base model class for Tensorflow-based model construction."""
from .data_source import BaseDataSource
import os
import time
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from .checkpoint_manager import CheckpointManager
import logging
logger = logging.getLogger(__name__)


class BaseModel(object):
    """Base model class for Tensorflow-based model construction.

    This class assumes that there exist no other Tensorflow models defined.
    That is, any variable that exists in the Python session will be grabbed by the class.
    """

    def __init__(self,
                 tensorflow_session: tf.Session,
                 train_data: BaseDataSource,
                 identifier: str = None):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._initialized = False
        self.__identifier = identifier

        self._data_format = self._train_data.data_format

        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # Log messages to file
        root_logger = logging.getLogger()

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Run-time parameters
        with tf.variable_scope('learning_params'):
            self.is_training = tf.placeholder(tf.bool)
            self.use_batch_statistics = tf.placeholder(tf.bool)

        self.output_tensors = self.build_model(self._train_data)
        logger.info('Built model.')

    @property
    def identifier(self):
        raise NotImplementedError

    @property
    def _identifier_suffix(self):
        """Identifier suffix for model based on data sources and parameters."""
        return ''

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def build_model(self, data_source: BaseDataSource):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize_if_not(self, training=False):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        with tf.variable_scope('savers'):
            self.checkpoint.build_savers()  # Create savers

        # Start pre-processing routines
        datasource = self._train_data
        datasource.preprocess_data()

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def inference_generator(self):
        """Perform inference on test data and yield a batch of output."""
        self.initialize_if_not(training=False)
        self.checkpoint.load_all()  # Load available weights

        data_source = self._train_data
        while True:
            data_source.preprocess_data()
            fetches = dict(self.output_tensors, **data_source.output_tensors)
            start_time = time.time()
            outputs = self._tensorflow_session.run(
                fetches=fetches,
                feed_dict={
                    self.is_training: False,
                    self.use_batch_statistics: True,
                },
            )
            outputs['inference_time'] = 1e3*(time.time() - start_time)
            yield outputs
