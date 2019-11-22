"""Data source of stream of frames."""
import bz2
import dlib
import queue
import shutil
import threading
import time
from typing import Tuple
import os
from urllib.request import urlopen

from collections import OrderedDict

import cv2 as cv
import numpy as np
import tensorflow as tf

from PIL import Image
from time import sleep


import logging
logger = logging.getLogger(__name__)

class FramesSource(object):
    """Preprocessing of stream of frames."""

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 batch_size: int,
                 eye_image_shape: Tuple[int, int],
                 data_format: str = 'NHWC',
                 **kwargs):
        """Create queues and threads to read and preprocess data."""

        # load image
        fn = "/home/jetson/wilfried/GazeML/src/test_imgs/Lenna.png"
        im = Image.open(fn)
        image = np.asanyarray(im)

        self.frame = image
        self._eye_image_shape = eye_image_shape
        self._proc_mutex = threading.Lock()

        self._last_frame_index = 0
        self._open = True

        assert tensorflow_session is not None and isinstance(tensorflow_session, tf.compat.v1.Session)
        assert isinstance(batch_size, int) and batch_size > 0
        self.data_format = data_format.upper()
        assert self.data_format == 'NHWC' or self.data_format == 'NCHW'
        self.batch_size = batch_size
        self._tensorflow_session = tensorflow_session
        self._coordinator = tf.train.Coordinator()

        # Setup file read queue
        self._fread_queue = queue.Queue(maxsize=batch_size)

        with tf.compat.v1.variable_scope(''.join(c for c in self.short_name if c.isalnum())):
            # Setup preprocess queue
            labels, dtypes, shapes = self._determine_dtypes_and_shapes()
            self._preprocess_queue = tf.FIFOQueue(
                    capacity=batch_size,
                    dtypes=dtypes, shapes=shapes,
            )
            self._tensors_to_enqueue = OrderedDict([
                (label, tf.placeholder(dtype, shape=shape, name=label))
                for label, dtype, shape in zip(labels, dtypes, shapes)
            ])

            self._enqueue_op = \
                self._preprocess_queue.enqueue(tuple(self._tensors_to_enqueue.values()))
            self._preprocess_queue_close_op = \
                self._preprocess_queue.close(cancel_pending_enqueues=True)
            self._preprocess_queue_size_op = self._preprocess_queue.size()
            self._preprocess_queue_clear_op = \
                self._preprocess_queue.dequeue_up_to(self._preprocess_queue.size())

            output_tensors = self._preprocess_queue.dequeue_many(self.batch_size)
            if not isinstance(output_tensors, list):
                output_tensors = [output_tensors]
            self._output_tensors = dict([
                (label, tensor) for label, tensor in zip(labels, output_tensors)
            ])

        logger.info('Initialized data source: "%s"' % self.short_name)


    _short_name = 'Single Frame'

    def __del__(self):
        """Destruct and clean up instance."""
        self.cleanup()

    __cleaned_up = False

    def cleanup(self):
        """Force-close all threads."""
        if self.__cleaned_up:
            return

        self._coordinator.request_stop()

        # Unblock any self._fread_queue.put calls
        while True:
            try:
                self._fread_queue.get_nowait()
            except queue.Empty:
                break
            time.sleep(0.1)

        # Push data through to trigger exits in preprocess/transfer threads
        for _ in range(self.batch_size):
            self._fread_queue.put(None)
        self._tensorflow_session.run(self._preprocess_queue_close_op)

        self._coordinator.join(self.all_threads, stop_grace_period_secs=5)
        self.__cleaned_up = True

    def _determine_dtypes_and_shapes(self):
        """Determine the dtypes and shapes of Tensorflow queue and staging area entries."""
        while True:
            raw_entry = next(self.entry_generator(yield_just_one=True))
            if raw_entry is None:
                continue
            preprocessed_entry_dict = self.preprocess_entry(raw_entry)
            if preprocessed_entry_dict is not None:
                break
        labels, values = zip(*list(preprocessed_entry_dict.items()))
        dtypes = [value.dtype for value in values]
        shapes = [value.shape for value in values]
        return labels, dtypes, shapes

    def read_entry_job(self):
        """Job to read an entry and enqueue to _fread_queue."""
        read_entry = self.entry_generator()
        entry = next(read_entry)
        if entry is not None:
            self._fread_queue.put(entry)

    def set_frame(self, frame):
        self.frame = frame

    def preprocess_data(self):
        self.read_entry_job()
        self.preprocess_job()

    def preprocess_job(self):
        """Job to fetch and preprocess an entry."""
        raw_entry = self._fread_queue.get()
        if raw_entry is None:
            return
        preprocessed_entry_dict = self.preprocess_entry(raw_entry)
        if preprocessed_entry_dict is not None:
            feed_dict = dict([(self._tensors_to_enqueue[label], value)
                              for label, value in preprocessed_entry_dict.items()])
            self._tensorflow_session.run(self._enqueue_op, feed_dict=feed_dict)

    @property
    def output_tensors(self):
        """Return tensors holding a preprocessed batch."""
        return self._output_tensors

    @property
    def short_name(self):
        """Short name specifying source."""
        return self._short_name

    def entry_generator(self, yield_just_one=False):
        """Generate eye image entries by detecting faces and facial landmarks."""
        try:
            while range(1) if yield_just_one else True:
                # Grab frame
                with self._proc_mutex:
                    bgr = self.frame
                    bgr = cv.flip(bgr, flipCode=1)  # Mirror
                    current_index = self._last_frame_index + 1
                    self._last_frame_index = current_index

                    grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                    frame = {
                        'frame_index': current_index,
                        'time': {
                            'before_frame_read': 0,
                            'after_frame_read': 0,
                        },
                        'bgr': bgr,
                        'grey': grey,
                    }

                # Eye image segmentation pipeline
                self.detect_faces(frame)
                self.detect_landmarks(frame)
                self.segment_eyes(frame)
                frame['time']['after_preprocessing'] = time.time()

                for i, eye_dict in enumerate(frame['eyes']):
                    yield {
                        'frame_index': np.int64(current_index),
                        'eye': eye_dict['image'],
                        'eye_index': np.uint8(i),
                    }

        finally:
            # Execute any cleanup operations as necessary
            pass

    def preprocess_entry(self, entry):
        """Preprocess segmented eye images for use as neural network input."""
        eye = entry['eye']
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self.data_format == 'NHWC' else 0)
        entry['eye'] = eye
        return entry

    def detect_faces(self, frame):
        """Detect all faces in a frame."""
        frame_index = frame['frame_index']

        detector = get_face_detector()
        if detector.__class__.__name__ == 'CascadeClassifier':
            detections = detector.detectMultiScale(frame['grey'])
        else:
            detections = detector(cv.resize(frame['grey'], (0, 0), fx=0.5, fy=0.5), 0)
        faces = []
        for d in detections:
            try:
                l, t, r, b = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
                l *= 2
                t *= 2
                r *= 2
                b *= 2
                w, h = r - l, b - t
            except AttributeError:  # Using OpenCV LBP detector on CPU
                l, t, w, h = d
            faces.append((l, t, w, h))
        faces.sort(key=lambda bbox: bbox[0])
        frame['faces'] = faces
        frame['last_face_detect_index'] = frame['frame_index']

    def detect_landmarks_68(self, frame):
        """Detect 68-point facial landmarks for faces in frame."""
        predictor = get_landmarks_68_predictor()
        landmarks = []
        for face in frame['faces']:
            l, t, w, h = face
            rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
            landmarks_dlib = predictor(frame['grey'], rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts
            landmarks.append(np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)]))

        # Project down to 5-point landmarks for compability
        five_point_landmarks = [landmarks[36]] + \
                               [landmarks[39]] + \
                               [landmarks[27]] + \
                               [landmarks[42]] + \
                               [landmarks[45]]
        frame['landmarks'] = five_point_landmarks


    def detect_landmarks(self, frame):
        """Detect 5-point facial landmarks for faces in frame."""
        predictor = get_landmarks_predictor()
        landmarks = []
        for face in frame['faces']:
            l, t, w, h = face
            rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
            landmarks_dlib = predictor(frame['grey'], rectangle)

            def tuple_from_dlib_shape(index):
                p = landmarks_dlib.part(index)
                return (p.x, p.y)

            num_landmarks = landmarks_dlib.num_parts
            landmarks.append(np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)]))
        frame['landmarks'] = landmarks

    def segment_eyes(self, frame):
        """From found landmarks in previous steps, segment eye image."""
        eyes = []

        # Final output dimensions
        oh, ow = self._eye_image_shape

        frame_landmarks = frame['landmarks']

        for face, landmarks in zip(frame['faces'], frame_landmarks):
            # Segment eyes
            # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
            for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
                x1, y1 = landmarks[corner1, :]
                x2, y2 = landmarks[corner2, :]
                eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
                if eye_width == 0.0:
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                # Centre image on middle of eye
                translate_mat = np.asmatrix(np.eye(3))
                translate_mat[:2, 2] = [[-cx], [-cy]]
                inv_translate_mat = np.asmatrix(np.eye(3))
                inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

                # Rotate to be upright
                roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
                rotate_mat = np.asmatrix(np.eye(3))
                cos = np.cos(-roll)
                sin = np.sin(-roll)
                rotate_mat[0, 0] = cos
                rotate_mat[0, 1] = -sin
                rotate_mat[1, 0] = sin
                rotate_mat[1, 1] = cos
                inv_rotate_mat = rotate_mat.T

                # Scale
                scale = ow / eye_width
                scale_mat = np.asmatrix(np.eye(3))
                scale_mat[0, 0] = scale_mat[1, 1] = scale
                inv_scale = 1.0 / scale
                inv_scale_mat = np.asmatrix(np.eye(3))
                inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

                # Centre image
                centre_mat = np.asmatrix(np.eye(3))
                centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
                inv_centre_mat = np.asmatrix(np.eye(3))
                inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

                # Get rotated and scaled, and segmented image
                transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
                inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                     inv_centre_mat)
                eye_image = cv.warpAffine(frame['grey'], transform_mat[:2, :], (ow, oh))
                if is_left:
                    eye_image = np.fliplr(eye_image)
                eyes.append({
                    'image': eye_image,
                    'inv_landmarks_transform_mat': inv_transform_mat,
                    'side': 'left' if is_left else 'right',
                })
        frame['eyes'] = eyes

_face_detector = None
_landmarks_predictor = None
_landmarks_68_predictor = None


def _get_dlib_data_file(dat_name):
    dat_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    dat_path = '%s/%s' % (dat_dir, dat_name)
    if not os.path.isdir(dat_dir):
        os.mkdir(dat_dir)

    # Download trained shape detector
    if not os.path.isfile(dat_path):
        with urlopen('http://dlib.net/files/%s.bz2' % dat_name) as response:
            with bz2.BZ2File(response) as bzf, open(dat_path, 'wb') as f:
                shutil.copyfileobj(bzf, f)

    return dat_path


def _get_opencv_xml(xml_name):
    xml_dir = os.path.relpath('%s/../3rdparty' % os.path.basename(__file__))
    xml_path = '%s/%s' % (xml_dir, xml_name)
    if not os.path.isdir(xml_dir):
        os.mkdir(xml_dir)

    # Download trained shape detector
    if not os.path.isfile(xml_path):
        url_stem = 'https://raw.githubusercontent.com/opencv/opencv/master/data/lbpcascades'
        with urlopen('%s/%s' % (url_stem, xml_name)) as response:
            with open(xml_path, 'wb') as f:
                shutil.copyfileobj(response, f)

    return xml_path


def get_face_detector():
    """Get a singleton dlib face detector."""
    global _face_detector
    if not _face_detector:
        try:
            dat_path = _get_dlib_data_file('mmod_human_face_detector.dat')
            _face_detector = dlib.cnn_face_detection_model_v1(dat_path)
        except:
            xml_path = _get_opencv_xml('lbpcascade_frontalface_improved.xml')
            _face_detector = cv.CascadeClassifier(xml_path)
    return _face_detector


def get_landmarks_predictor():
    """Get a singleton dlib face 5 points landmark predictor."""
    global _landmarks_predictor
    if not _landmarks_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_5_face_landmarks.dat')
        # dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_predictor

def get_landmarks_68_predictor():
    """Get a singleton dlib face 68 points landmark predictor."""
    global _landmarks_68_predictor
    if not _landmarks_68_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_68_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_68_predictor
