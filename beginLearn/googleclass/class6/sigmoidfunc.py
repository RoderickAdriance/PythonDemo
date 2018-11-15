import math

def basic_sigmoid(x):
    """
       Compute sigmoid of x.

       Arguments:
       x -- A scalar

       Return:
       s -- sigmoid(x)
       """
    v = 1 / (1+math.exp(-x))
    return v


v = basic_sigmoid(1)
print(v)
