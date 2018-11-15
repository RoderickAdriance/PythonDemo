import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from exercise.c13.kt_utils import *

import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

def HappyModel(input_shape):
    X_Input=Input(input_shape)

    X=ZeroPadding2D((3,3))(X_Input)

    X=Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)

    X=BatchNormalization(axis=3,name='bn0')(X)

    X=Activation('relu')(X)

    X=MaxPooling2D((2,2),name='max_pool')(X)

    X=Flatten()(X)

    X=Dense(1,activation='sigmoid',name='fc')(X)

    model=Model(inputs=X_Input,outputs =X,name='HappyModel')

    return model


def run_model():
    happyModel = HappyModel(X_train.shape[1:])
    happyModel.compile(optimizer="Adam", loss="binary_crossentropy", metrics=['accuracy'])
    happyModel.fit(x=X_train, y=Y_train, epochs=50, batch_size=64)
    preds = happyModel.evaluate(x=X_test, y=Y_test)
    print()
    print("Loss = " + str(preds[0]))
    print("PythonDemo Accuracy = "+str(preds[1]))

    # 非常关键的保存模型
    happyModel.save('happy_v1.h5')
    happyModel.summary()


    # plot_model(happyModel, to_file='HappyModel.png')
    # SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))



run_model()