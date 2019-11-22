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
        batch_size = 1

        # Define frame data source
        # Change data_format='NHWC' if not using CUDA

        print("Create Data source")
        data_source = FramesSource(eye_image_shape=(36, 60))

        print("Create Data source finished")
        print("Create Model")
        model = ELG(
                    session, data_source=data_source,
                    data_format='NCHW' if gpu_available else 'NHWC',
                    batch_size = batch_size
                )
        print("Create Model finished")

        print ("Start inference")
        output = model.inference()
        print("inference finished")
        print("Print output:")
        print(output)
