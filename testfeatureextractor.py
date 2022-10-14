import tflite_runtime.interpreter as tflite
import numpy as np
import pandas as pd
import os

l = []
CSV = 'newface4.csv'
folder = os.listdir('testdata/')


def get_mobilenet_input(f, out_size=(92, 112), is_quant=True):
    img = np.array(Image.open("testdata/" + f).resize(out_size))
    if not (is_quant):
        img = img.astype(np.float32) / 128 - 1
    return np.array([img])



model_file = 'mobnetfacefeatureExtractor.tflite'
interpreter = tflite.Interpreter(model_path=model_file)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
##Converting image into tensor

import cv2
from PIL import Image
import numpy as np
#import tensorflow as tf

for f in folder:
    print(f)
    image = cv2.resize(cv2.imread('testdata/' + f).astype(np.float32), (224, 224))
    img = get_mobilenet_input(f)

    ##Test model
    interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    features = np.array(output_data)
    features = np.ravel(features)
    l.append(features)
    print(features.shape)

import csv

l = zip(*l)
with open(CSV, 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file

    writer.writerow(folder)
    for row in l:
        writer.writerow(row)
