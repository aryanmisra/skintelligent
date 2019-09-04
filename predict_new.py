from tensorflow import keras
import numpy as np
import model
import main
import PIL
from PIL import Image
import argparse
import time
import os
import sys
import json
from os import listdir
from os.path import isfile, join
t0 = time.time()
    
print("Loading model..")
model_sec_a = keras.models.load_model('saves/model_A.h5', compile=False)

model_sec_c = keras.models.load_model('saves/model_C.h5', compile=False)

model_sec_p = keras.models.load_model('saves/model_P.h5', compile=False)

model_sec_u = keras.models.load_model('saves/model_U.h5', compile=False)

model_sec_w = keras.models.load_model('saves/model_W.h5', compile=False)

t1 = time.time()
print("Models loaded in {0:.5f} seconds.".format(t1-t0))
im_paths = [f for f in listdir('set_valid/') if isfile(join('set_valid/', f))]
im_paths.sort()
try:    
    for i in im_paths:
        im_path = 'set_valid/'+i
        print(im_path)
        #print("Loading image..")
        img = Image.open(im_path)
        img = img.resize((main.input_x,main.input_y), Image.ANTIALIAS)
        img = np.array(img) / 255.0
        img = np.reshape(img,[1,main.input_x,main.input_y,3])
        #print("done image..")
        #print("predicting model..")
        class_a = model_sec_a.predict(img)
        class_c = model_sec_c.predict(img)
        class_p = model_sec_p.predict(img)
        class_u = model_sec_u.predict(img)
        class_w = model_sec_w.predict(img)
        print(np.argmax(np.squeeze(class_a)), end=" ")
        print(np.argmax(np.squeeze(class_c)),end=" ")
        print(np.argmax(np.squeeze(class_p)),end=" ")
        print(np.argmax(np.squeeze(class_u)),end=" ")
        print(np.argmax(np.squeeze(class_w)))        
        
        
except KeyboardInterrupt:
    sys.exit()