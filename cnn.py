import numpy
import cv2
import PIL
import os, os.path
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dropout
from keras import callbacks
from keras.layers import Dense, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD,Adam
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras import backend as k

k.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
trainblur_directory = './s_cnn/train/blur/'
trainnoblur_directory = './s_cnn/train/no_blur/'
testblur_directory = './s_cnn/test/0/blur/'
testnoblur_directory = './s_cnn/test/0/no_blur/'
filepath = "./s_cnn/models/"

num_classes = 2
#########################################################
#loading blurry images
img_data_list1=[]
data_dir_list1 = os.listdir(trainblur_directory)

img_list1=os.listdir(trainblur_directory)
for img in img_list1:
	input_img=cv2.imread(trainblur_directory +  img )
	input_img=numpy.swapaxes(input_img,0,2)

	img_data_list1.append(input_img)

img_data1 = numpy.array(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 /= 255

print(img_data1.shape)
num_of_samples1 = img_data1.shape[0]
labels1 = numpy.ones((num_of_samples1,),dtype='int64')
print("length of labels1 is "+str(len(labels1)))
print("labels1 are all "+str(labels1[10]))

##########################################################
#laoding none blurry images

img_data_list2=[]
data_dir_list2 = os.listdir(trainnoblur_directory)

img_list2=os.listdir(trainnoblur_directory)
for img in img_list2:
	input_img=cv2.imread(trainnoblur_directory +  img )
	input_img = numpy.swapaxes(input_img, 0, 2)
	img_data_list2.append(input_img)

img_data2 = numpy.array(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 /= 255
print(img_data2.shape)

num_of_samples2 = img_data2.shape[0]
labels2 = numpy.ones((num_of_samples2,),dtype='int64')
labels2[:]=0
print("length of labels2 is "+str(len(labels2)))
print("labels1 are all "+str(labels2[10]))
#######################################################
# Combine the two numpy arrays and shuffle
labels=numpy.concatenate((labels1,labels2),axis=0)
img_data = numpy.concatenate((img_data1,img_data2),axis=0)
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
###########################################################
# Defining the model
input_shape=img_data[0].shape

model = Sequential()
model.add(Convolution2D(96, 7,7,input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(256, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

###########################
epochs = 100
learning_rate = 0.01
decay = learning_rate / epochs
adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
numpy.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

# Training
hist = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=1, validation_data=(X_test, y_test))

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# tensorboard callback
tensorboard_callback = k.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

callbacks_list = [csv_log,early_stopping,checkpoint, tensorboard_callback]


# Evaluating the model

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

# Save our model here
file = open(filepath+"motionblur.h5", 'a')
model.save(filepath+"motionblur.h5")
file.close()
######################################################

