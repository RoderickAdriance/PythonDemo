import numpy as np

def image2vector(image):
    print(image.shape[0])
    print(image.shape[1])
    print(image.shape[2])

    #重置为18行一列的矩阵
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    print(v)
    return v

image = np.array([
       [[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

image2vector(image)