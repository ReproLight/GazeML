import numpy as np
import tensorflow as tf
import os

from core.summary_manager import SummaryManager

class ELG(object):
    _hg_first_layer_stride = 1
    _hg_num_modules = 2
    _hg_num_feature_maps = 32
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 train_data = {},
                 test_data = {},
                 test_losses_or_metrics: str = None,
                 use_batch_statistics_at_test: bool = True,
                 identifier: str = None):
        """Initialize model with data sources and parameters."""
        self._tensorflow_session = tensorflow_session
        #self._train_data = train_data
        self._test_data = test_data
        self._test_losses_or_metrics = None
        self._initialized = False
        #self.__identifier = identifier

        # Extract and keep known prefixes/scopes
        self._learning_schedule = [
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ]
        self._known_prefixes = [schedule for schedule in self._learning_schedule]

        # Check consistency of given data sources
        ## FIXME
        # train_data_sources = list(train_data.values())
        # test_data_sources = list(test_data.values())
        # all_data_sources = train_data_sources + test_data_sources
        # first_data_source = all_data_sources.pop()

        self._batch_size = 2
        self._data_format = "NCHW" # when using gpu "NHWC"

        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        ## FIXME
        #if not os.path.isdir(self.output_path):
        #    os.makedirs(self.output_path)
        #
        # Log messages to file
        # root_logger = logging.getLogger()
        # file_handler = logging.FileHandler(self.output_path + '/messages.log')
        # file_handler.setFormatter(root_logger.handlers[0].formatter)
        # for handler in root_logger.handlers[1:]:  # all except stdout
        #     root_logger.removeHandler(handler)
        # root_logger.addHandler(file_handler)
        #
        # Register a manager for tf.Summary
        self.summary = SummaryManager(self)
        #
        # # Register a manager for checkpoints
        # self.checkpoint = CheckpointManager(self)
        #
        # # Register a manager for timing related operations
        # self.time = TimeManager(self)
        #
        # # Prepare for live (concurrent) validation/testing during training, on the CPU
        # self._enable_live_testing = (len(self._train_data) > 0) and (len(self._test_data) > 0)
        # self._tester = LiveTester(self, self._test_data, use_batch_statistics_at_test)
        #
        # Run-time parameters
        with tf.compat.v1.variable_scope('learning_params'):
            self.is_training = tf.compat.v1.placeholder(tf.bool)
            self.use_batch_statistics = tf.compat.v1.placeholder(tf.bool)
            self.learning_rate_multiplier = tf.Variable(1.0, trainable=False, dtype=tf.float32)
            self.learning_rate_multiplier_placeholder = tf.compat.v1.placeholder(dtype=tf.float32)
            self.assign_learning_rate_multiplier = \
                tf.compat.v1.assign(self.learning_rate_multiplier, self.learning_rate_multiplier_placeholder)

        self._build_all_models()

    def _build_all_models(self):
        """Build training (GPU/CPU) and testing (CPU) streams."""
        self.output_tensors = {}
        self.loss_terms = {}
        self.metrics = {}

        def _build_datasource_summaries(data_sources):
            """Register summary operations for input data from given data sources."""
            with tf.compat.v1.variable_scope('test_data'):
                for data_source_name, data_source in data_sources.items():
                    tensors = data_source.output_tensors
                    for key, tensor in tensors.items():
                        summary_name = '%s/%s' % (data_source_name, key)
                        shape = tensor.shape.as_list()
                        num_dims = len(shape)
                        if num_dims == 4:  # Image data
                            if shape[1] == 1 or shape[1] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_first')
                            elif shape[3] == 1 or shape[3] == 3:
                                self.summary.image(summary_name, tensor,
                                                   data_format='channels_last')
                            # TODO: fix issue with no summary otherwise
                        elif num_dims == 2:
                            self.summary.histogram(summary_name, tensor)
                        else:
                            print('I do not know how to create a summary for %s (%s)' %
                                         (summary_name, tensor.shape.as_list()))

        def _build_test():
            data_sources = self._test_data

            # Build model
            output_tensors, loss_terms, metrics = self.build_model(data_sources)

            # Record important tensors
            self.output_tensors = output_tensors
            self.loss_terms = loss_terms
            self.metrics = metrics

        # If there are any test data streams, build same model with different scope
        # Trainable parameters will be copied at test time
        _build_datasource_summaries(self._test_data)
        with tf.compat.v1.variable_scope('test'):
            _build_test()

    def build_model(self, data_sources):
        """Build model."""
        #data_source = next(iter(data_sources.values()))
        #input_tensors = data_source.output_tensors
        #x = input_tensors['eye']
        x = tf.compat.v1.placeholder(tf.float32, shape=(2, 1, 36, 60), name='eye')

        with tf.compat.v1.variable_scope('input_data'):
            self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)

        outputs = {}
        loss_terms = {}
        metrics = {}

        with tf.compat.v1.variable_scope('hourglass'):
            # Prepare for Hourglass by downscaling via conv
            with tf.compat.v1.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(x, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

        #     # Hourglass blocks
        #     x_prev = x
        #     for i in range(self._hg_num_modules):
        #         with tf.variable_scope('hg_%d' % (i + 1)):
        #             x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
        #             x, h = self._build_hourglass_after(
        #                 x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
        #             )
        #             self.summary.feature_maps('hmap%d' % i, h, data_format=self._data_format_longer)
        #             if y1 is not None:
        #                 metrics['heatmap%d_mse' % (i + 1)] = _tf_mse(h, y1)
        #             x_prev = x
        #     if y1 is not None:
        #         loss_terms['heatmaps_mse'] = tf.reduce_mean([
        #             metrics['heatmap%d_mse' % (i + 1)] for i in range(self._hg_num_modules)
        #         ])
        #     x = h
        #     outputs['heatmaps'] = x
        #
        # # Soft-argmax
        # x = self._calculate_landmarks(x)
        # with tf.variable_scope('upscale'):
        #     # Upscale since heatmaps are half-scale of original image
        #     x *= self._hg_first_layer_stride
        #     if y2 is not None:
        #         metrics['landmarks_mse'] = _tf_mse(x, y2)
        #     outputs['landmarks'] = x
        #
        # # Fully-connected layers for radius regression
        # with tf.variable_scope('radius'):
        #     x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
        #     for i in range(3):
        #         with tf.variable_scope('fc%d' % (i + 1)):
        #             x = tf.nn.relu(self._apply_bn(self._apply_fc(x, 100)))
        #     with tf.variable_scope('out'):
        #         x = self._apply_fc(x, 1)
        #     outputs['radius'] = x
        #     if y3 is not None:
        #         metrics['radius_mse'] = _tf_mse(tf.reshape(x, [-1]), y3)
        #         loss_terms['radius_mse'] = 1e-7 * metrics['radius_mse']
        #     self.summary.histogram('radius', x)

        # Define outputs
        return outputs, loss_terms, metrics

    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.compat.v1.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.compat.v1.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.compat.v1.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.compat.v1.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.compat.v1.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x