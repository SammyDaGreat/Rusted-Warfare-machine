import numpy as np
import math
import time

def f(x,y):
    if x == 0:
        return math.copysign(0.25,y)
    if y == 0:
        return 0.25-math.copysign(0.25,x)
    z = math.atan(y/x)
    z = z/math.pi/2
    if x<0 and y>0:
        z = 0.5+z
    elif x<0 and y<0:
        z = z-0.5
    return z

def cord2alpha(cords):
    x,y = cords[0],cords[1]
    x = x-288
    y = -y+476
    alpha = f(x,y)
    if alpha<0:
        alpha = 1+alpha
    return alpha

def standardize(that_uuid):
    try:
        i=0
        while(True):
                time.sleep(0.01)
                cords = np.load('Data//y1//'+str(that_uuid)+'.'+str(i)+'.npy')
                alpha = np.array(cord2alpha(cords))
                np.save('Data//y1//'+str(that_uuid)+'.'+str(i)+'.npy',alpha)
                i += 1
    except:
        return