import os
from SimpleCV import *
import matplotlib.pyplot as plt
import time
from pylab import imshow, figure, show, subplot
import numpy as np
import Image
import cv2

# Lectura de imagenes


def rho(v):

	path = '/home/pi/lunares/imagenes seleccionadas'

	listimg = [os.path.join(dirpath,f)
       		for dirpath, dirnames, files in os.walk(path)
        	 for f in files if f.endswith('.png')]

	#img = np.zeros(len(listimg))


	#for t in xrange(38):
  	img          = cv2.imread(listimg[v])
	dimY,dimX    = img.shape[:2]
 	b,g,r        = cv2.split(img)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Binarizar

	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        p = np.array([140, 132, 89, 65, 65, 145, 18, 32, 32, 100, 135, 100, 50, 50, 50, 50, 18, 108, 20, 95, 95, 95, 40, 71, 20, 90, 38, 38, 38, 25, 90, 90, 95, 80, 80, 75])
        #print p[v]
        #print listimg[v]
	ret, thresh1 = cv2.threshold(img_gray,p[v],255,cv2.THRESH_BINARY)
        #thresh1 = cv2.adaptiveThreshold(img_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        #cv2.THRESH_BINARY,11,2)
	im_array_b = np.array(thresh1)


#cv2.imshow('image',thresh1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###### Algorithm edges detector

	A_r = 0
	A_g = 0
	A_b = 0
	B_r = 0
	B_g = 0
	B_b = 0
	contN = 0
	contL = 0

### Promedio de una imagen lunar/no lunar
        
	for i in xrange(dimY):
  	  for j in xrange(dimX):
    	    if im_array_b[i][j] == 0:
      		A_r = r[i][j] + A_r  	
      		A_g = g[i][j] + A_g
      		A_b = b[i][j] + A_b
      		contL = contL +1
    	    else:
      		B_r = r[i][j] + B_r
      		B_g = g[i][j] + B_g
      		B_b = b[i][j] + B_b
      		contN = contN +1

       
	A_r = float(A_r)/float(contL)
	A_g = float(A_g)/float(contL)
	A_b = float(A_b)/float(contL)
	B_r = float(B_r)/float(contN)
	B_g = float(B_g)/float(contN)
	B_b = float(B_b)/float(contN)

	rho = [float(A_r)/float(B_r), float(A_r)/float(B_g), float(A_r)/float(B_b), float(A_g)/float(B_r),float(A_g)/float(B_g), float(A_g)/float(B_b), float(A_b)/float(B_r), float(A_b)/float(B_g), float(A_b)/float(B_b)]

#print rho
	return rho
