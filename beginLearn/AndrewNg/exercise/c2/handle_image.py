from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
from scipy import misc
import random
from exercise.c2 import lr_utils

def handle(pic_name):
    im = Image.open("pic/"+pic_name)
    pic_arr = np.array(im)
    return pic_arr

def showpic(index):
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()
    picture=train_set_x_orig[index]
    plt.imshow(picture)
    plt.show()
    pic_name = int(random.uniform(1, 2000))
    misc.imsave("pic/"+str(pic_name)+".jpg",picture)

# for i in range(209):
#     showpic(i)
