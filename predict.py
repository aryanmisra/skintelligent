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
t0 = time.time()
    
print("Loading model..")
model = keras.models.load_model('saves/model_N_23.h5')

model_sec_a = keras.models.load_model('saves/backup/model_A_1.h5', compile=False)

model_sec_c = keras.models.load_model('saves/backup/model_C_6.h5', compile=False)

model_sec_p = keras.models.load_model('saves/backup/model_P_5.h5', compile=False)

model_sec_u = keras.models.load_model('saves/backup/model_U_5.h5', compile=False)

model_sec_w = keras.models.load_model('saves/backup/model_W_6.h5', compile=False)

t1 = time.time()
print("Models loaded in {0:.5f} seconds.".format(t1-t0))
try:
    # while 1:
    #im_path = input("Image path: ")
    print("Loading image..")
    # img = Image.open('images/%s' % im_path)
    img = Image.open('../images/acne/3_right.png')
    img = img.resize((main.input_x,main.input_y), Image.ANTIALIAS)
    img = np.array(img) / 255.0
    img = np.reshape(img,[1,main.input_x,main.input_y,3])
    print("done image..")
    print("predicting model..")

    classes = model.predict(img)
    print(classes)
    classes[classes < 0.2] = 0
    classes[classes >= 0.2] = 1
    out = [
        0,
        0,
        0,
        0,
        0
    ]
    classes = np.squeeze(classes)
    
    if classes[0] == 1:
        classes_sec_a = model_sec_a.predict(img)
        idx_a = np.argmax(classes_sec_a)
        if idx_a == 0:
            out[0] = 2
        elif idx_a == 1:
            out[0] = 3
    if classes[1] == 1:
        
        classes_sec_c = model_sec_c.predict(img)
        idx_c = np.argmax(classes_sec_c)
        if idx_c == 0:
            out[1] = 2
        elif idx_c == 1:
            out[1] = 3
    if classes[2] == 1:
        
        classes_sec_p = model_sec_p.predict(img)
        idx_p = np.argmax(classes_sec_p)
        if idx_p == 0:
            out[2] = 2
        elif idx_p == 1:
            out[2] = 3

    if classes[3] == 1:
        
        classes_sec_u = model_sec_u.predict(img)
        idx_u = np.argmax(classes_sec_u)
        if idx_u == 0:
            out[3] = 2
        elif idx_u == 1:
            out[3] = 3
            
    if classes[4] == 1:
        
        classes_sec_w = model_sec_w.predict(img)
        idx_w = np.argmax(classes_sec_w)
        if idx_w == 0:
            out[4] = 2
        elif idx_c == 1:
            out[4] = 3

    for x in range(len(out)):
        print(out[x])
        if out[x] == 0:
            out[x] = 1

    out_dict = {"labels": [
        {"label": "A", "intensity": out[0]}, 
        {"label": "C", "intensity": out[1]}, 
        {"label": "P", "intensity": out[2]}, 
        {"label": "U", "intensity": out[3]}, 
        {"label": "W", "intensity": out[4]}]}

    # with open('result_%s.json' % im_path, 'w') as fp:
    #     json.dump(out_dict, fp)
except KeyboardInterrupt:
    sys.exit()