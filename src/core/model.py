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
                 learning_schedule: List[Dict[str, Any]] = [],
                 train_data: Dict[str, BaseDataSource] = {},
                 test_data: Dict[str, BaseDataSource] = {},
                 identifier: str = None):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        self._train_data = train_data
        self._test_data = test_data
        self._initialized = False
        self.__identifier = identifier

        # Extract and keep known prefixes/scopes
        self._learning_schedule = learning_schedule
        self._known_prefixes = [schedule for schedule in learning_schedule]

        # Check consistency of given data sources
        train_data_sources = list(train_data.values())
        test_data_sources = list(test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        first_data_source = all_data_sources.pop()
        self._batch_size = first_data_source.batch_size
        self._data_format = first_data_source.data_format
        for data_source in all_data_sources:
            if data_source.batch_size != self._batch_size:
                raise ValueError(('Data source "%s" has anomalous batch size of %d ' +
                                  'when detected batch size is %d.') % (data_source.short_name,
                                                                        data_source.batch_size,
                                                                        self._batch_size))
            if data_source.data_format != self._data_format:
                raise ValueError(('Data source "%s" has anomalous data_format of %s ' +
                                  'when detected data_format is %s.') % (data_source.short_name,
                                                                         data_source.data_format,
                                                                         self._data_format))
        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        # Log messages to file
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(self.output_path + '/messages.log')
        file_handler.setFormatter(root_logger.handlers[0].formatter)
        for handler in root_logger.handlers[1:]:  # all except stdout
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)

        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)

        # Run-time parameters
        with tf.variable_scope('learning_params'):
            self.is_training = tf.placeholder(tf.bool)
            self.use_batch_statistics = tf.placeholder(tf.bool)
            self.learning_rate_multiplier = tf.Variable(1.0, trainable=False, dtype=tf.float32)
            self.learning_rate_multiplier_placeholder = tf.placeholder(dtype=tf.float32)
            self.assign_learning_rate_multiplier = \
                tf.assign(self.learning_rate_multiplier, self.learning_rate_multiplier_placeholder)

        self._build_all_models()

    def __del__(self):
        """Explicitly call methods to cleanup any live threads."""
        train_data_sources = list(self._train_data.values())
        test_data_sources = list(self._test_data.values())
        all_data_sources = train_data_sources + test_data_sources
        for data_source in all_data_sources:
            data_source.cleanup()

    __identifier_stem = None

    @property
    def identifier(self):
        """Identifier for model based on time."""
        if self.__identifier is not None:  # If loading from checkpoints or having naming enforced
            return self.__identifier
        if self.__identifier_stem is None:
            self.__identifier_stem = self.__class__.__name__ + '/' + time.strftime('%y%m%d%H%M%S')
        return self.__identifier_stem + self._identifier_suffix

    @property
    def _identifier_suffix(self):
        """Identifier suffix for model based on data sources and parameters."""
        return ''

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}

        def _build_train_or_test(mode):
            data_sources = self._train_data if mode == 'train' else self._test_data

            # Build model
            output_tensors, loss_terms, metrics = self.build_model(data_sources, mode=mode)

            # Record important tensors
            self.output_tensors[mode] = output_tensors
            self.loss_terms[mode] = loss_terms
            self.metrics[mode] = metrics

        # Build the main model
        if len(self._train_data) > 0:
            _build_train_or_test(mode='train')
            logger.info('Built model.')

            # Print no. of parameters and lops
            flops = tf.profiler.profile(
                options=tf.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.float_operation()
                ).with_empty_output().build())
            logger.info('------------------------------')
            logger.info(' Approximate Model Statistics ')
            logger.info('------------------------------')
            logger.info('FLOPS per input: {:,}'.format(flops.total_float_ops / self._batch_size))
            logger.info(
                'Trainable Parameters: {:,}'.format(
                    np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
                )
            )
            logger.info('------------------------------')

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        if len(self._test_data) > 0:
            with tf.variable_scope('test'):
                _build_train_or_test(mode='test')
            logger.info('Built model for live testing.')

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
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
        for _, datasource in self._train_data.items():
            datasource.preprocess_data()

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def inference_generator(self):
        """Perform inference on test data and yield a batch of output."""
        self.initialize_if_not(training=False)
        self.checkpoint.load_all()  # Load available weights

        # TODO: Make more generic by not picking first source
        data_source = next(iter(self._train_data.values()))
        while True:
            data_source.preprocess_data()
            fetches = dict(self.output_tensors['train'], **data_source.output_tensors)
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
