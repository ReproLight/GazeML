import numpy as np
import tensorflow as tf
import os
import time

from core.summary_manager import SummaryManager
from core.checkpoint_manager import CheckpointManager

class ELG(object):
    _hg_first_layer_stride = 1
    _hg_num_modules = 2
    _hg_num_feature_maps = 32
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1

    _softargmax_coords = None

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
        self._data_format = "NHWC" # GPU "NCHW"

        self._data_format_longer = ('channels_first' if self._data_format == 'NCHW'
                                    else 'channels_last')

        # Make output dir
        ## FIXME
        if not os.path.isdir(self.output_path):
            raise ValueError(f"model weight path not found: {self.output_path}")
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
        # Register a manager for checkpoints
        self.checkpoint = CheckpointManager(self)
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

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        eye_image_shape = (36, 60)
        eh, ew = eye_image_shape

        return 'ELG_i%dx%d_f%dx%d_n%d_m%d' % (
            ew, eh,
            int(ew / self._hg_first_layer_stride),
            int(eh / self._hg_first_layer_stride),
            self._hg_num_feature_maps, self._hg_num_modules,
        )

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
        #with tf.compat.v1.variable_scope('test'):
        _build_test()

    def build_model(self, data_sources):
        """Build model."""
        #data_source = next(iter(data_sources.values()))
        #input_tensors = data_source.output_tensors
        #x = input_tensors['eye']

        outputs = {}
        loss_terms = {}
        metrics = {}

        eye_tensor = tf.compat.v1.placeholder(tf.float32, shape=(self._batch_size, 1, 36, 60), name='eye')
        outputs['eye'] = eye_tensor
        x = eye_tensor

        with tf.compat.v1.variable_scope('input_data'):
            self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)

        with tf.compat.v1.variable_scope('hourglass'):
            # Prepare for Hourglass by downscaling via conv
            with tf.compat.v1.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(x, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

            # Hourglass blocks
            x_prev = x
            for i in range(self._hg_num_modules):
                with tf.compat.v1.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, h = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    self.summary.feature_maps('hmap%d' % i, h, data_format=self._data_format_longer)
                    x_prev = x
            x = h
            outputs['heatmaps'] = x

        # Soft-argmax
        x = self._calculate_landmarks(x)
        with tf.compat.v1.variable_scope('upscale'):
            # Upscale since heatmaps are half-scale of original image
            x *= self._hg_first_layer_stride
            outputs['landmarks'] = x

        # Fully-connected layers for radius regression
        with tf.compat.v1.variable_scope('radius'):
            x = tf.contrib.layers.flatten(tf.transpose(x, perm=[0, 2, 1]))
            for i in range(3):
                with tf.compat.v1.variable_scope('fc%d' % (i + 1)):
                    x = tf.nn.relu(self._apply_bn(self._apply_fc(x, 100)))
            with tf.compat.v1.variable_scope('out'):
                x = self._apply_fc(x, 1)
            outputs['radius'] = x
            self.summary.histogram('radius', x)

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

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

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

    def _build_hourglass(self, x, steps_to_go, num_features, depth=1):
        with tf.compat.v1.variable_scope('depth%d' % depth):
            # Upper branch
            up1 = x
            for i in range(self._hg_num_residual_blocks):
                up1 = self._build_residual_block(up1, num_features, num_features,
                                                 name='up1_%d' % (i + 1))
            # Lower branch
            low1 = self._apply_pool(x, kernel_size=2, stride=2)
            for i in range(self._hg_num_residual_blocks):
                low1 = self._build_residual_block(low1, num_features, num_features,
                                                  name='low1_%d' % (i + 1))
            # Recursive
            low2 = None
            if steps_to_go > 1:
                low2 = self._build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
            else:
                low2 = low1
                for i in range(self._hg_num_residual_blocks):
                    low2 = self._build_residual_block(low2, num_features, num_features,
                                                      name='low2_%d' % (i + 1))
            # Additional residual blocks
            low3 = low2
            for i in range(self._hg_num_residual_blocks):
                low3 = self._build_residual_block(low3, num_features, num_features,
                                                  name='low3_%d' % (i + 1))
            # Upsample
            if self._data_format == 'NCHW':  # convert to NHWC
                low3 = tf.transpose(low3, (0, 2, 3, 1))
            up2 = tf.compat.v1.image.resize_bilinear(
                    low3,
                    up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                    align_corners=True,
                  )
            if self._data_format == 'NCHW':  # convert back from NHWC
                up2 = tf.transpose(up2, (0, 3, 1, 2))

        return up1 + up2

    def _build_hourglass_after(self, x_prev, x_now, do_merge=True):
        with tf.compat.v1.variable_scope('after'):
            for j in range(self._hg_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._hg_num_feature_maps,
                                                   self._hg_num_feature_maps,
                                                   name='after_hg_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.compat.v1.variable_scope('hmap'):
                h = self._apply_conv(x_now, self._hg_num_landmarks, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.compat.v1.variable_scope('merge'):
                with tf.compat.v1.variable_scope('h'):
                    x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.compat.v1.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_hmaps
        return x_next, h

    def _calculate_landmarks(self, x):
        """Estimate landmark location from heatmaps."""
        with tf.compat.v1.variable_scope('argsoftmax'):
            if self._data_format == 'NHWC':
                _, h, w, _ = x.shape.as_list()
            else:
                _, _, h, w = x.shape.as_list()
            if self._softargmax_coords is None:
                # Assume normalized coordinate [0, 1] for numeric stability
                ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                             np.linspace(0, 1.0, num=h, endpoint=True),
                                             indexing='xy')
                ref_xs = np.reshape(ref_xs, [-1, h*w])
                ref_ys = np.reshape(ref_ys, [-1, h*w])
                self._softargmax_coords = (
                    tf.constant(ref_xs, dtype=tf.float32),
                    tf.constant(ref_ys, dtype=tf.float32),
                )
            ref_xs, ref_ys = self._softargmax_coords

            # Assuming N x 18 x 45 x 75 (NCHW)
            beta = 1e2
            if self._data_format == 'NHWC':
                x = tf.transpose(x, (0, 3, 1, 2))
            x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
            x = tf.nn.softmax(beta * x, axis=-1)
            lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[2])
            lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[2])

            # Return to actual coordinates ranges
            return tf.stack([
                lmrk_xs * (w - 1.0) + 0.5,
                lmrk_ys * (h - 1.0) + 0.5,
            ], axis=2)  # N x 18 x 2

    def initialize_if_not(self):
        """Initialize variables and begin preprocessing threads."""
        if self._initialized:
            return

        # Build supporting operations
        with tf.variable_scope('savers'):
            self.checkpoint.build_savers()  # Create savers

        # Initialize all variables
        self._tensorflow_session.run(tf.global_variables_initializer())
        self._initialized = True

    def inference(self, input):
        """Perform inference on test data and yield a batch of output."""
        self.initialize_if_not()
        self.checkpoint.load_all()  # Load available weights

        #fetches = dict(self.output_tensors, **input_tensors)
        dict(self.output_tensors)
        start_time = time.time()
        outputs = self._tensorflow_session.run(
            fetches=self.output_tensors,
            feed_dict={
                self.output_tensors['eye']: input['eye'],
                self.is_training: False,
                self.use_batch_statistics: True,
            },
        )
        #outputs['inference_time'] = 1e3 * (time.time() - start_time)
        return outputs