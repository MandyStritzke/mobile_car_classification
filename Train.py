#!/usr/bin/env python
# coding: utf-8

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time # temporary for computation time analysis


import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from glob import glob
import random
import numpy as np

import pylib as py
import pickle
from tensorflow.keras.optimizers import SGD
import pandas
import math

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import precision_score, accuracy_score , confusion_matrix

import datetime


#tf.autograph.set_verbosity(0)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# arguments from command line
py.arg('--epochs', type=int, default = 20)
py.arg('--lr', type=float, default = 0.001)
py.arg('--opt', default='adam', choices=['adam', 'sgd'])
args = py.args()

nr_epochs = 20#args.epochs
lr = 0.001 #args.lr
 

IMG_SIZE = 224

batch_size = 32

VAL_SPLIT = 0.1

n_channels = 3

#pandas.set_option('display.max_rows', 100)

data_dir = "./car_data/car_data/" #local file path


#### logs
log_dir = "./logs/run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1 , write_grads = True, write_graph = False)

def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=n_channels)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image

def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder = True)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def make_dataset(path, batch_size, test = None):
    classes = [f for f in sorted(os.listdir(path)) if not f.startswith('.')] # retrieve classes from subdir, ignoring hidden files
    #sorted, s.t. the folders are in alphabetical order (same order in train and val is important for one-hot-encoding)
    
    filenames = glob(path + '/*/*.jpg')
    
    n_data = len(filenames)
    print("Anzahl Bilder:", n_data)
    train_size = math.floor((1-VAL_SPLIT)*n_data)
    print("Anzahl Training:", train_size)
    val_data = n_data - train_size
    
    
    if test is None:
    # no shuffling in testset for better evaluation of each image
        random.shuffle(filenames)
   
    labels = [classes.index(name.split('/')[-2]) for name in filenames] # returning class index
    #labels = [name.split('/')[-2] for name in filenames] # returning string labels

    classes = np.array(classes)
    
    labels = tf.one_hot(labels,len(classes), dtype = tf.int8) # encode class index labels
    

    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)

    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    if test is None:
    # create dataset for validation
        ds = tf.data.Dataset.zip((images_ds, labels_ds))
        data = ds.take(train_size)
        data = configure_for_performance(data)
        
        val_data = ds.skip(train_size)
        val_data = configure_for_performance(val_data)
     
    else:
        # for the testset: include filenames
        ds = tf.data.Dataset.zip((images_ds, labels_ds, filenames_ds))
        data = ds
        val_data = None
        
      
    return data, val_data , classes, train_size


from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization


def resblock(input, num_channels, filter):
    x = input
    y = Conv2D(num_channels, kernel_size = filter , padding = 'same')(input)
    y = BatchNormalization()(y)
    y = relu(y)
    y = Conv2D(num_channels, kernel_size = filter , padding = 'same')(y)
    y = BatchNormalization()(y)
    residual = Add()([y, x])
    x = relu(residual)
    return x


def build_model():

   # initializer = tf.keras.initializers.HeNormal()
    img_input = tf.keras.Input(shape = (IMG_SIZE,IMG_SIZE,n_channels))
    
    lay1 = tf.keras.layers.experimental.preprocessing.Rescaling( 1.0/255.0 )(img_input)
    lay1 = tf.keras.layers.experimental.preprocessing.RandomRotation( 1, fill_mode='wrap' )(lay1)
    #lay1 = tf.keras.layers.experimental.preprocessing.RandomFlip()(lay1)
    #lay1 = tf.keras.layers.experimental.preprocessing.RandomZoom( (-0, -0.5) )(lay1)
    
    #lay1 = tf.keras.layers.GaussianNoise(0.1)(lay1)
    

    conv1 = tf.keras.layers.Conv2D(16,(3,3), input_shape = (IMG_SIZE,IMG_SIZE,n_channels), activation ='relu', strides=(1,1), padding = 'same')(lay1) #128 x 128 x 16
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size =(2, 2))(conv1)#64x64x16

    res1 = resblock(pool1, 16,(3,3))

    conv2 = tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', strides=(1,1), padding='same')(res1) #64x64x32
    #conv2 = BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size =(2, 2))(conv2)

    res2 = resblock(pool2, 32, (3,3))
    pool3 = tf.keras.layers.MaxPooling2D(pool_size =(2, 2))(res2)

    conv3 = tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', strides=(1,1), padding='same')(pool3) 
    #conv3 = BatchNormalization()(conv3)
    res3 = resblock(conv3,64,(3,3))

    flatt = tf.keras.layers.Flatten()(res3)
    dense1 = tf.keras.layers.Dense(128, activation = 'relu')(flatt)

    out = tf.keras.layers.Dense(n_classes,activation ='softmax')(dense1)

    model = tf.keras.Model(inputs=img_input, outputs= out)
    return model



tic = time.perf_counter()
train_dataset ,val_dataset , classes , train_size = make_dataset((data_dir + 'train'), batch_size)
toc = time.perf_counter()
print(f"Took {toc-tic:0.4f} seconds to load the dataset.")
n_classes = len(classes)


#model = build_model()
from tensorflow.keras.applications.resnet50 import ResNet50

img_input = tf.keras.Input(shape = (IMG_SIZE,IMG_SIZE,n_channels))

lay1 = tf.keras.layers.experimental.preprocessing.Rescaling( 1.0/255.0 )(img_input)
#lay1 = tf.keras.layers.experimental.preprocessing.RandomRotation( 1, fill_mode='wrap' )(lay1)
##lay1 = tf.keras.layers.experimental.preprocessing.RandomFlip()(lay1)
##lay1 = tf.keras.layers.experimental.preprocessing.RandomZoom( (-0, -0.5) )(lay1)
#lay1 = tf.keras.layers.GaussianNoise(0.1)(lay1)



model = ResNet50(weights = None, classes = n_classes, include_top = False, input_shape = (IMG_SIZE, IMG_SIZE,n_channels), pooling = 'max')

output_tensor = model(lay1)

fc = tf.keras.layers.Dense(1028, activation = 'relu')(output_tensor)
fc = tf.keras.layers.Dense(n_classes, activation ='softmax')(fc)
model = tf.keras.Model(inputs = img_input, outputs= fc)


print(model.summary())

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def get_opt(opt, lr):
    if opt == 'adam':
        return tf.keras.optimizers.Adam(lr = lr)
    elif opt == 'sgd':
        return SGD(lr = lr)


optimizer = get_opt(args.opt, args.lr)
lr_metric = get_lr_metric(optimizer=optimizer)

model.compile( optimizer=optimizer,
               loss=tf.losses.CategoricalCrossentropy(from_logits=False),
               metrics=['accuracy', lr_metric]
               )
# sparse categorical loss is for index encoded data, categorical for one-hot encoded data


# steps per epoch: needed because the input-data is repeating indefinitely, so we need to tell when one epoch is finished
steps_epochs = math.floor(train_size/batch_size)

history = model.fit( train_dataset, validation_data = val_dataset, validation_steps = math.floor(steps_epochs*VAL_SPLIT),
                     epochs=nr_epochs,
                    steps_per_epoch = steps_epochs, verbose = 2  , callbacks=[tensorboard_callback]
                )
                
                
#-----
# read test data (first in tf.dataset and then back into np.array)
test_data = make_dataset((data_dir+'test'), batch_size, test = "0")[0]

batched_dataset = test_data.batch(1000)

iterator = iter(batched_dataset)
next_element = iterator.get_next()

X_test = np.array(next_element[0])
Y_test = np.array(next_element[1])
filenames_test = (np.array(next_element[2])).astype(str) #convert 'object' type to string type
#filenames_test = filenames_test.astype(str)
filenames_test = np.array([x.split('/') for x in filenames_test])
filenames_test = filenames_test[:,-1] # only get filename (last column)
#-----

Y_test_label = np.argmax(Y_test, axis = 1) #true label

print("Predict ...")
Y_pred = model.predict(X_test)
Y_pred_prop = np.around(np.max(Y_pred, axis = 1),3)
Y_pred_label = np.argmax(Y_pred, axis = 1)

#for manual validation: show filenames, predscore etc. for looking up the pictures
overview = (np.vstack((filenames_test, Y_test_label, Y_pred_label, Y_pred_prop))).T
df = pandas.DataFrame(data = overview , columns = ['img_name', 'true_class', 'pred_class', 'label_prop'])

# sort for instances with wrong label (acc = 0) and high prop.
df['acc'] = (df['true_class'] == df['pred_class']).astype(int)
df = df.sort_values(['acc','label_prop'],ascending = [True,False])
print(df)




#alternative accuracy with numpy:
acc = np.mean(Y_test_label== Y_pred_label)

print("Accuracy: ", acc)

#acc = accuracy_score(Y_test_label, Y_pred_label)


#conf matrix with pandas:
conf_matrix = pandas.crosstab(pandas.Series(Y_test_label,name ='Actual'),pandas.Series(Y_pred_label,name = 'Predicted'))


#conf_matrix = confusion_matrix(Y_test_label,Y_pred_label)
print(conf_matrix)



# save model
tf.keras.models.save_model(model,'model.hp5',save_format ='h5')

# save history
with open('trainhistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Model saved.")

#
## Plot history: loss + accuracy
#
#f, ax = plt.subplots(1,2)# , sharey= True)
#
#ax[0].plot(history.history['loss'], label='training data')
#ax[0].plot(history.history['val_loss'], label='validation data')
#ax[0].set_title('Loss')
#
#
#ax[1].plot(history.history['accuracy'], label='training data')
#ax[1].plot(history.history['val_accuracy'], label='validation data')
#ax[1].set_title('Accuracy')
#
#for axe in ax.flat:
#    axe.set(xlabel='No. epoch', ylabel='y-label')
#
## Hide x labels and tick labels for top plots and y ticks for right plots.
#for axe in ax.flat:
#    axe.label_outer()
#plt.legend(loc="upper left")
#plt.show()
