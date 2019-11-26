#!/usr/bin/env python3
"""Main script for gaze direction inference from webcam feed."""
import argparse
import os
import queue
import threading
import time

import coloredlogs
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image


from datasources import FramesSource
from models import ELG
import util.gaze


if __name__ == '__main__':
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level="INFO",
    )

    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(session_config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2
        big_model = False

        print("Create Data source")
        if big_model:
            eye_image_shape = (108, 180)
            data_source = FramesSource(eye_image_shape=eye_image_shape)
        else:
            eye_image_shape = (36, 60)
            data_source = FramesSource(eye_image_shape=eye_image_shape)

        print("Create Data source finished")
        print("Create Model")
        if big_model:
            model = ELG(
                session,
                data_format='NCHW' if gpu_available else 'NHWC',
                batch_size=batch_size,
                first_layer_stride=3,
                num_modules=3,
                num_feature_maps=64,
                eye_image_shape=eye_image_shape
            )
        else:
            model = ELG(
                        session,
                        data_format='NCHW' if gpu_available else 'NHWC',
                        batch_size=batch_size,
                        eye_image_shape=eye_image_shape
                    )

        print("Create Model finished")

        print("Start inference")
        eyes = data_source.entry_generator()
        output = model.inference(eyes)
        print("inference finished")
        print("Print output:")
        print(output)
