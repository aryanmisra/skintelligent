import json
import numpy as np
import os
import time
import tensorflow as tf
#utility package for main train script
#parsing jsons into array format

json_path = '../json/'
img_dir = '../images/'

def fileList(source): #creating a list of all json files in directory
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.json')):
                matches.append(os.path.join(root, filename))
    return matches

json_list = fileList(json_path)

#ran into a bit of an effiecency problem due to all jsons not having labels in same order
#now we have to iterate through each json, looking for specific labels to drop into the array, rather than directly extracting from inside the json to array
def json_process(constraint):
    count = 0
    Y = np.zeros([len(json_list), 5])
    for i in range(len(json_list)):
        with open (json_list[i]) as json_data:
            annotation = json.loads(json_data.read())
            temp_fn = "../images/"+annotation['filename']
            del annotation['filename']
            print(temp_fn)
            if os.path.exists(temp_fn):
                for x in annotation:
                    for j in range(0,5):
                        if annotation[x][j]['label'] == 'A':
                            Y[i-count,0] = annotation[x][j]['intensity']
                        if annotation[x][j]['label'] == 'C':
                            Y[i-count,1] = annotation[x][j]['intensity']
                        if annotation[x][j]['label'] == 'P':
                            Y[i-count,2] = annotation[x][j]['intensity']
                        if annotation[x][j]['label'] == 'U':
                            Y[i-count,3] = annotation[x][j]['intensity']
                        if annotation[x][j]['label'] == 'W':
                            Y[i-count,4] = annotation[x][j]['intensity'] 
            else:
                Y = np.delete(Y,(i-count),axis=0)
                count +=1
    if constraint:
        Y[Y == 1] = 0
        Y[Y == 2] = 1
        Y[Y == 3] = 1
        #Y = (tf.cast(Y, tf.int64))# - 1.0) / 3.0
        return Y
    else:
        Y[Y == 1] = 0
        Y[Y == 2] = 2
        Y[Y == 3] = 3
        #Y = (tf.cast(Y, tf.int64))# - 1.0) / 3.0
        return Y

def json_process_se(cat):

    Y = json_process(False)
    paths = image_paths()
    print(Y.shape)
    print(len(paths))
    new_P = []
    if cat == 'A':
        Y = np.delete(Y, [1,2,3,4], axis=1)
    elif cat == 'C':
        Y = np.delete(Y, [0,2,3,4], axis=1)
    elif cat == 'P':
        Y = np.delete(Y, [0,1,3,4], axis=1)
    elif cat == 'U':
        Y = np.delete(Y, [0,1,2,4], axis=1)
    elif cat == 'W':
        Y = np.delete(Y, [0,1,2,3], axis=1)
    else:
        raise Exception('This function should only recieve an input of either A, C, P, U, or W')

    new_Y = np.zeros((len(Y), 2))
    for i in range(len(Y)):
        if Y[i,0] == 2:
            new_Y[i,0] = 0
            print(i)
            new_P.append(paths[i])
        if Y[i,0] == 3:
            print(i)
            new_Y[i,1] = 1
            new_P.append(paths[i])
    new_Y = new_Y[~np.all(new_Y == 0, axis=1)]
    return new_Y, new_P
        
def length(cat):
    Y = json_process(True)
    if cat == 'A':
        Y = np.delete(Y, [1,2,3,4], axis=1)
        return len(Y[~np.all(Y == 0, axis=1)])
    elif cat == 'C':
        Y = np.delete(Y, [0,2,3,4], axis=1)
        return len(Y[~np.all(Y == 0, axis=1)])
    elif cat == 'P':
        Y = np.delete(Y, [0,1,3,4], axis=1)
        return len(Y[~np.all(Y == 0, axis=1)])
    elif cat == 'U':
        Y = np.delete(Y, [0,1,2,4], axis=1)
        return len(Y[~np.all(Y == 0, axis=1)])
    elif cat == 'W':
        Y = np.delete(Y, [0,1,2,3], axis=1)
        return len(Y[~np.all(Y == 0, axis=1)])
    else:
        raise Exception('This function should only recieve an input of either A, C, P, U, or W')


def image_paths():
    paths = []
    for w in range(len(json_list)):
        with open (json_list[w]) as json_data:
            annotation = json.loads(json_data.read())
            del annotation['labels']
            #paths.append(annotation['filename'].split('/', 1)[-1])
            temp_fn = "../images/"+annotation['filename']
            if os.path.exists(temp_fn):
                paths.append(annotation['filename'])
    return paths

def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
            
def rotate(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def flip(x: tf.Tensor) -> tf.Tensor:
    
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x: tf.Tensor) -> tf.Tensor:
    
    #x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def zoom(x: tf.Tensor) -> tf.Tensor:
    
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        crops = tf.image.crop_and_resize([tf.squeeze(img)], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(224,224))
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

if __name__ == '__main__':
    print("\nUtility script for parsing json into feature array\n")
    t0 = time.time()
    json_process_se('C')
    t1 = time.time()
    print("Process completed in {0:.5f} seconds for {1} files.".format(t1-t0, len(json_list)))