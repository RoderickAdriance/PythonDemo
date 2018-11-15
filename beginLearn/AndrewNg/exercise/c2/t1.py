import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from exercise.c2 import lr_utils

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

def showimage(index):
    print(train_set_x_orig[index].shape)
    plt.imshow(train_set_x_orig[index])
    plt.show()
    print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")

showimage(1)

# print(train_set_x_orig[0].shape[1])   #第1个图片的'列'维度
# print(train_set_y.T.shape)    #转置

#将 64x64x3 折叠为209,12288     (图片数,每张图片的纬度数)
# train_set_x_flatten = train_set_x_orig.reshape(209,-1)
# test_set_x_flatten = test_set_x_orig.reshape(50, -1)
#
# train_set_x = train_set_x_flatten/255
# test_set_x  = test_set_x_flatten/255
# print("train_set_x:",train_set_x.shape)
# print("test_set_x:",test_set_x)

