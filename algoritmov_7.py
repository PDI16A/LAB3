import os
from SimpleCV import *
import matplotlib.pyplot as plt
import time
from pylab import imshow, figure, show, subplot
import numpy as np
import Image
import cv2
from rho import rho
from rho_v_h import rho_v_h 

####   Cargar la Imagen 
c            = Camera()
time.sleep(2)
c.live()
disp = Display()

while disp.isNotDone():
    img = c.getImage()
    img.save("fotito.png")
    
    if disp.mouseLeft:
        break
    break

print 1
img          = cv2.imread('fotito.png')
#img           = cv2.imread('foto24.png') #foto24
b,g,r        = cv2.split(img)
dimY,dimX     = img.shape[:2]
img = img[160:480, 120:360]
#img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print 2
####    RHO
i = 0
promedio_rho = np.zeros(9)

#for i in xrange (35):
  ##print rho(i)/37
  #promedio_rho = [x / 35 for x in rho(i)]+promedio_rho
  ##print promedio_rho

#print promedio_rho
#promedio_rho = np.zeros((1,9))

#promedio_rho = np.array([ 0.88514457, 0.91865718, 0.951379, 0.81332341, 0.8395462, 0.86523403, 0.79292307, 0.81673789, 0.83794959])

#promedio_rho = np.array([0.92840856, 1.3580664, 1.49985772, 0.72389686, 0.84920741, 0.91193558, 0.69712877, 0.80772624, 0.84455017])

#promedio_rho = np.array([0.92421243, 1.40108677, 1.58200915, 0.65852912, 0.82488949, 0.89552386, 0.63000456, 0.77670443, 0.81708553])

promedio_rho = [0.62172421, 0.86772321, 0.98658823, 0.33588284, 0.40824325, 0.44232023, 0.32626412, 0.40587053, 0.44513191]

print promedio_rho
#### Mascara


imgout = rho_v_h(img, promedio_rho)

#out13 = img.edges(t1=150)
#out14 = img.sobel(2, 0, True, 7)
#out15 = img.sobel(0, 2, True, 7)
#out16 = img.prewitt()


########## PLOT
from matplotlib import pyplot as plt

img_rgb = cv2.merge([r,g,b]) 
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(imgout, 'gray'), plt.title('Algoritmo del profesor')
plt.xticks([]), plt.yticks([])
plt.show()

#plt.subplot(121), plt.imshow(img_rgb), plt.title('Original')
#plt.xticks([]), plt.yticks([])
#plt.subplot(122), plt.imshow(imgout, 'gray'), plt.title('otra')
#plt.xticks([]), plt.yticks([])
#plt.subplot(221), plt.imshow(out13, 'gray'), plt.title('canny')
#plt.xticks([]), plt.yticks([])
#plt.subplot(222), plt.imshow(out16, 'gray'), plt.title('prewitt')
#plt.xticks([]), plt.yticks([])
#plt.show()



