#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

from standalone.model import ELG

fn = "./test_imgs/left_eye.png"
im = Image.open(fn)
eye = np.asanyarray(im)
eye_scaled = eye[::3,::3] / 255.0
input_frame = eye_scaled[np.newaxis, ...]
input_frame = input_frame.astype(np.float32)
input_batch = np.stack([input_frame,input_frame], axis=0)
#input_tensor = tf.convert_to_tensor(input_frame, dtype=tf.float32)
input = {'eye': input_batch}

with tf.compat.v1.Session() as session:
    model = ELG(session)
    output = model.inference(input)

    with open("gazeml_standalone_network.txt", "w") as f:
        graph = tf.get_default_graph()
        operations = graph.get_operations()
        for op in operations:
            name = op.name
            inputs = op.inputs
            outputs = op.outputs
            print(f"name:    {name}", file=f)
            print(f"outputs: {outputs}", file=f)
            print(40*"-", file=f)
    import sys
    sys.exit(0)
    from IPython import embed
    embed()