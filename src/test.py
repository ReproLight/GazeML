import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

from standalone.model import ELG

fn = "./test_imgs/left_eye.png"
im = Image.open(fn)
eye = np.asanyarray(im)
eye_scaled = eye[::3,::3] / 255.0
input_frame = eye_scaled[np.newaxis, np.newaxis, ...]
input_frame = input_frame.astype(np.float32)
#input_tensor = tf.convert_to_tensor(input_frame, dtype=tf.float32)
input = {'eye': input_frame}

with tf.compat.v1.Session() as session:
    model = ELG(session)
    output = model.inference(input)

    from IPython import embed
    embed()