import tensorflow as tf
from tensorflow import keras
from utils import image_paths, json_process, json_process_se
import matplotlib.pyplot as plt
import utils
import time
import os
import argparse
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import CNN_model, CNN_model_sec
AUTOTUNE = tf.data.experimental.AUTOTUNE
input_x = 300
input_y = 300
batch_size = 32
epochs = 500
augmentations = [utils.flip, utils.color, utils.zoom, utils.rotate]
parser = argparse.ArgumentParser()
parser.add_argument("--cont", '-c', help="resume training",
                    action="store_true")
parser.add_argument("--train",'-t',type=str, help="choose what to train")
def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()
def visualization(history,mode):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('vis/model_acc_%s.png' % mode)
    plt.clf()
    time.sleep(1)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('vis/model_loss_%s.png' % mode)
    
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [input_x, input_y])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_load_image(path):
    final_im_path = utils.img_dir + path
    image = tf.io.read_file(final_im_path)
    return preprocess_image(image)
def maprange(x):
    return tf.clip_by_value(x, 0, 1)
def train(mode):
    t0 = time.time()
    if mode == 'N':
        image_path_list = image_paths()
        labels = json_process(True)
    elif mode == 'A':
        labels, image_path_list = json_process_se('A')
    elif mode == 'C':
        labels, image_path_list = json_process_se('C')
    elif mode == 'P':
        labels, image_path_list = json_process_se('P')
    elif mode == 'U':
        labels, image_path_list = json_process_se('U')
    elif mode == 'W':
        labels, image_path_list = json_process_se('W')

    #train_size = int(0.8 * len(labels))
    #val_size = int(0.2 * len(labels))

    path_ds = tf.data.Dataset.from_tensor_slices(image_path_list)
    image_ds = path_ds.map(preprocess_load_image, num_parallel_calls=AUTOTUNE)
    
    for f in augmentations:
        
        image_ds = image_ds.map(f, num_parallel_calls=4)
    
    image_ds = image_ds.map(maprange, num_parallel_calls=4)
    
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    ds = image_label_ds.shuffle(buffer_size=1000)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    v_ds = ds.take(int(0.25 * len(labels))) 
    t_ds = ds.skip(int(0.25 * len(labels)))

    
    t1 = time.time()
    print("Preprocessing completed in {0:.5f} seconds.".format(t1-t0))
    if mode == 'N':
        net = CNN_model(input_x,input_y,3)
    else:
        net = CNN_model_sec(input_x, input_y, 3)

    logdir = os.path.join(
        "logs",
        "fit",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    #os.mkdir(logdir)
    callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', 
        min_delta=1e-6, 
        patience=10,
        verbose=1),
    keras.callbacks.ModelCheckpoint(
        filepath=('saves/model_%s_{epoch}.h5' % (mode)),
        save_best_only=True,
        monitor='val_loss',
        verbose=1),
    #keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
        patience=2, min_lr=0.00001)
    ]
    
    print(net.summary())
    steps_per_epoch=tf.math.ceil(len(image_path_list)/batch_size).numpy()
    t0 = time.time()
    history = net.fit(t_ds, epochs=epochs,callbacks=callbacks, steps_per_epoch=steps_per_epoch, validation_data=v_ds, verbose=1)
    t1 = time.time()
    print("Training completed in {0:.5f} seconds.".format(t1-t0))

    visualization(history, mode)


if __name__ == '__main__':
    args = parser.parse_args()
    """
    if args.cont:
        train(True)
    else:
        train(False)
    """
    
    if args.train == 'A':
        train('A')
    elif args.train == 'C':
        train('C')
    elif args.train == 'P':
        train('P')
    elif args.train == 'U':
        train('U')
    elif args.train == 'W':
        train('W')
    else:
        train('N')