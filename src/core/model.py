"""Base model class for Tensorflow-based model construction."""
from datasources import FramesSource
import os
import time

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
                 data_source: FramesSource):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        self._data_source = data_source
        self._initialized = False

        self._data_format = self._data_source.data_format

        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        if not os.path.isdir(self.output_path):
            raise ValueError(f"model weights not found at {self.output_path}")

        # Run-time parameters
        with tf.variable_scope('learning_params'):
            self.is_training = tf.placeholder(tf.bool)
            self.use_batch_statistics = tf.placeholder(tf.bool)

        with tf.compat.v1.variable_scope("SingleFrame"):
            # Setup preprocess queue
            self._preprocess_queue = tf.FIFOQueue(
                    capacity=self._data_source.batch_size,
                    dtypes=[tf.float32], shapes=[self._data_source.input_shape],
            )
            self.input_tensor = tf.placeholder(tf.float32, shape=self._data_source.input_shape, name='eye')

        self.output_tensors = self.build_model()
        logger.info('Built model.')
        self.initialize()
        logger.info('Initialized model.')
        logger.info('Weights loaded')

    @property
    def identifier(self):
        raise NotImplementedError

    @property
    def output_path(self):
        """Path to store logs and model weights into."""
        return '%s/%s' % (os.path.abspath(os.path.dirname(__file__) + '/../../outputs'),
                          self.identifier)

    def build_model(self):
        """Build model."""
        raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def initialize(self):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Register a manager for checkpoints
        checkpoint = CheckpointManager(self)

        # Build supporting operations
        with tf.variable_scope('savers'):
            checkpoint.build_savers()  # Create savers

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        checkpoint.load_all()
        self._initialized = True

    def inference(self):
        """Perform inference on test data and yield a batch of output."""
        eye = self._data_source.preprocess_data()
        self._tensorflow_session.run(self._preprocess_queue.enqueue([self.input_tensor]),
                                     feed_dict={self.input_tensor: eye})
        start_time = time.time()
        outputs = self._tensorflow_session.run(
            fetches=self.output_tensors,
            feed_dict={
                self.is_training: False,
                self.use_batch_statistics: True,
            },
        )
        outputs['inference_time'] = time.time() - start_time
        return outputs
