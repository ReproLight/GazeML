"""Single frame data source for gaze estimation."""
import cv2 as cv

from .frames import FramesSource
from PIL import Image
import numpy as np
from time import sleep

class SingleFrame(FramesSource):
    """Webcam frame grabbing and preprocessing."""

    def __init__(self, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Single frame'

        # load image
        fn = "./test_imgs/Lenna.png"
        im = Image.open(fn)
        image = np.asanyarray(im)

        self.frame = image
        # Call parent class constructor
        super().__init__(**kwargs)

    def set_frame(self, frame):
        self.frame = frame

    def frame_generator(self):
        """yield frame"""
        while True:
            yield self.frame
            sleep(1)
