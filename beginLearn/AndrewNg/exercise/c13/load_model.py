from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from exercise.c13.kt_utils import *
from keras.applications.imagenet_utils import preprocess_input

model = load_model('happy_v1.h5')
model.summary()

for i in range(1, 9):
    print(i)
    img_path = 'images/' + str(i) + '.jpg'
    img = image.load_img(img_path, target_size=(64, 64))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print(model.predict(x))

