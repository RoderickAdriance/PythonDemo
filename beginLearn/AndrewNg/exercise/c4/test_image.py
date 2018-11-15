from PIL import Image
import numpy as np
from pylab import *

# 读取图像
im = Image.open("pic/2.jpg")
# im.show()

# 原图像缩放
im_resized = im.resize((230, 230))
# im_resized.show()

arr = np.array(im_resized)
arr2 = arr.reshape(1, -1)

print(arr.shape)
print(arr2.shape)
