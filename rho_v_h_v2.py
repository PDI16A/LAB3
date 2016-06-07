import os
from SimpleCV import *
import matplotlib.pyplot as plt
import time
from pylab import imshow, figure, show, subplot
import numpy as np
import Image
import cv2
import math


def rho_v_h(img, promedio_rho):
    b,g,r        = cv2.split(img)
    dimX,dimY    = img.shape[:2]
    
    A_h_r = float(0.1) 
    A_h_g = float(0.1)
    A_h_b = float(0.1)
    B_h_r = float(0.1)
    B_h_g = float(0.1)
    B_h_b = float(0.1)

    A_v_r = float(0.1)
    A_v_g = float(0.1)
    A_v_b = float(0.1)
    B_v_r = float(0.1)
    B_v_g = float(0.1)
    B_v_b = float(0.1)
    

    rho_h = np.zeros((1,9))
    rho_v = np.zeros((1,9))

    N_h = np.zeros((dimX, dimY))
    N_v = np.zeros((dimX, dimY))

    rho_h1 = np.ones((1,9), dtype = float)
    rho_v1 = np.ones((1,9), dtype = float)

    N_h1 = np.zeros((dimX, dimY))
    N_v1 = np.zeros((dimX, dimY))
    
    imgout = np.zeros((dimX, dimY))
    
    f      = 1

    r = r + f
    g = g + f
    b = b + f

    for i in xrange(1,dimX-1):
        for j in xrange(1,dimY-1):
            A_h_r = r[i-1][j]+ 0.1 
            A_h_g = g[i-1][j]+ 0.1 
            A_h_b = b[i-1][j]+ 0.1
  
            B_h_r = r[i+1][j] + 0.1
            B_h_g = g[i+1][j] + 0.1
            B_h_b = b[i+1][j]+ 0.1
 
            A_v_r = r[i][j-1]+ 0.1 
            A_v_g = g[i][j-1] + 0.1
            A_v_b = b[i][j-1]+ 0.1

            B_v_r = r[i][j+1] + 0.1
            B_v_g = g[i][j+1] + 0.1
            B_v_b = b[i][j+1]+ 0.1

            rho_h = np.array([float(A_h_r)/float(B_h_r), float(A_h_r)/float(B_h_g), float(A_h_r)/float(B_h_b), float(A_h_g)/float(B_h_r),float(A_h_g)/float(B_h_g), float(A_h_g)/float(B_h_b), float(A_h_b)/float(B_h_r), float(A_h_b)/float(B_h_g), float(A_h_b)/float(B_h_b)], dtype = float)
            rho_v = np.array([float(A_v_r)/float(B_v_r), float(A_v_r)/float(B_v_g), float(A_v_r)/float(B_v_b), float(A_v_g)/float(B_v_r),float(A_v_g)/float(B_v_g), float(A_v_g)/float(B_v_b), float(A_v_b)/float(B_v_r), float(A_v_b)/float(B_v_g), float(A_v_b)/float(B_v_b)], dtype = float)

           
            norm_h = np.linalg.norm(rho_h - promedio_rho)
            norm_v = np.linalg.norm(rho_v - promedio_rho)

            N_h[i][j] = norm_h
            N_v[i][j] = norm_v

##    for i in xrange(1,dimX-1):
##        for j in xrange(1,dimY-1):
##            A_h_r = r[i+1][j] 
##            A_h_g = g[i+1][j] 
##            A_h_b = b[i+1][j]  
##
##            B_h_r = r[i-1][j] 
##            B_h_g = g[i-1][j] 
##            B_h_b = b[i-1][j]
##
##            A_v_r = r[i][j+1] 
##            A_v_g = g[i][j+1] 
##            A_v_b = b[i][j+1]  
##
##            B_v_r = r[i][j-1] 
##            B_v_g = g[i][j-1] 
##            B_v_b = b[i][j-1]
##
##            rho_h1 = np.array([float(A_h_r)/float(B_h_r), float(A_h_r)/float(B_h_g), float(A_h_r)/float(B_h_b), float(A_h_g)/float(B_h_r),float(A_h_g)/float(B_h_g), float(A_h_g)/float(B_h_b), float(A_h_b)/float(B_h_r), float(A_h_b)/float(B_h_g), float(A_h_b)/float(B_h_b)], dtype = float)
##            rho_v1 = np.array([float(A_v_r)/float(B_v_r), float(A_v_r)/float(B_v_g), float(A_v_r)/float(B_v_b), float(A_v_g)/float(B_v_r),float(A_v_g)/float(B_v_g), float(A_v_g)/float(B_v_b), float(A_v_b)/float(B_v_r), float(A_v_b)/float(B_v_g), float(A_v_b)/float(B_v_b)], dtype = float)
##
##           
##            norm_h1 = np.linalg.norm(rho_h1 - promedio_rho)
##            norm_v1 = np.linalg.norm(rho_v1 - promedio_rho)
##
##            N_h1[i][j] = norm_h1
##            N_v1[i][j] = norm_v1



    ##k = 0.96 # foto24
   # if N_h.max()< N_h1.max():
   #     e_h = k*abs(N_h1.max()- np.min(N_h1[np.nonzero(N_h)]))
   #     e_v = k*abs(N_v1.max()- np.min(N_v1[np.nonzero(N_v)]))
   # else:
   #     e_h = k*abs(N_h.max()- np.min(N_h[np.nonzero(N_h)]))
   #     e_v = k*abs(N_v.max()- np.min(N_v[np.nonzero(N_v)]))
            
    k = 0.009 #0.009
    #if N_h.max()< N_h1.max():
    e_h = k*(N_h.max()+0.4) #- (np.min(N_h[np.nonzero(N_h)])*100 ))
    e_v = k*(N_v.max()+0.4)#- (np.min(N_v[np.nonzero(N_v)])*100 ))
    #else:
    #    e_h = k*abs(N_h1.max()) #- (np.min(N_h1[np.nonzero(N_h)])*100 ))
    #    e_v = k*abs(N_v1.max()) #- (np.min(N_v1[np.nonzero(N_v)])*100 ))
        
    print k
    print N_h.max()
    #print N_h[271][319]
    print np.min(N_h[np.nonzero(N_h)])
    print N_v
    print e_h
    print e_v
    #print N_h[329][413]


    for i in xrange(1,dimX-1):
        for j in xrange(1, dimY-1):


            if N_h[i][j] < e_h :
             imgout[i][j] = 255
            elif N_v[i][j] < e_v:
             imgout[i][j] = 255
             
            #if (N_h[i][j] < e_h) or (N_h1[i][j] < e_h):
            # imgout[i][j] = 255
            #elif (N_v[i][j] < e_v) or (N_v1[i][j] < e_v):
            # imgout[i][j] = 255

            #if N_h1[i][j] < e_h:
             #imgout[i][j] = 255
            #elif N_v1[i][j] < e_v:
             #imgout[i][j] = 255

    
    print imgout        
    return imgout
