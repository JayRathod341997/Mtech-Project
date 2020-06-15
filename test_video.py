# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:27:40 2020

@author: Jay Rathod
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def histogram_patch(img):
    return np.histogram(img, bins=10,range=(0,256))[0]
#    return cv2.calcHist([img],[0],None,[256],[0,256])

def getPatches(img_arr,M,N):
        tiles = [img_arr[x:x+M,y:y+N] for x in range(0,len(img_arr),M) for y in range(0,len(img_arr),N)]
        tiles=np.array(tiles)       
        hist_list = [histogram_patch(i) for i in tiles]
#        return hist_list
        return np.array(hist_list).ravel()




filename1 ='gmm_avg_face_model.sav'
gmm = pickle.load(open(filename1, 'rb'))
print("model loaded" )



path = '\\'
filename = 'test_1.avi'




k=100


total=0
frames=[]
gray_frame=[]
k_vector=[]


font = cv2.FONT_HERSHEY_SIMPLEX 
org = (20, 30) 
fontScale = 1
color = (0, 0, 0) 
thickness = 2 


cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video file")  
else :
    print("Open successfully" )

while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret == True: 
        frames.append(frame)
        cv2.imshow('img',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame.append(gray)
        total =total+1

        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
    else:   
    	break
 
cap.release() 
cv2.destroyAllWindows() 
print("Total frames = ",total)



#n=20


#
for i in range(0,total-k+1):
    gray = np.array(gray_frame[i:i+k+1]).sum(axis=0)//k
    M , N = gray.shape
    t=getPatches(gray,M//3,N//3)
    x_test = np.array(t).ravel()
#    cv2.imshow('Frame', frames[i:i+k])
    plt.imshow(np.array(frames[i:i+k+1]).sum(axis=0) // k)
#    plt.imshow(x_test.reshape(M,N))
    plt.show()
    value = gmm.score([x_test])
    print(i," : ", i+k,"--> ",value )
    print("Sigmoid value ",sigmoid(value))
