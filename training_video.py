# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:29:25 2020

@author: Jay Rathod
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import pickle

def histogram_patch(img):
    return np.histogram(img, bins=10,range=(0,256))[0]
#    return cv2.calcHist([img],[0],None,[256],[0,256])

def getPatches(img_arr,M,N):
        tiles = [img_arr[x:x+M,y:y+N] for x in range(0,len(img_arr),M) for y in range(0,len(img_arr),N)]
        tiles=np.array(tiles)       
        hist_list = [histogram_patch(i) for i in tiles]
#        return hist_list
        return np.array(hist_list).ravel()



path_filename = path+filename


total=0
frames=[]
gray_frame=[]
collection_of_frames=[] 


k_vector=[]
n_vector=[]

path = 'C:/Users/Jay Rathod/Videos//'
filename = 'Train.avi'

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
    print("Error opening video file")  
else :
    print("Open successfully" )



k=30


#font = cv2.FONT_HERSHEY_SIMPLEX 
#org = (20, 30) 
#fontScale = 1
#color = (0, 0, 0) 
#thickness = 2  

while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret == True: 
        cv2.imshow('img',frame)
        frames.append(frame) 
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


for i in range(total-k+1):
    avg_frame=np.sum(gray_frame[i:i+k+1],axis=0)//k
    M,N = avg_frame.shape
    patch=getPatches(avg_frame,M//3,N//3)
    n_vector.append(patch)


X_train = np.array(n_vector)

print("Model is Training.......")

#print("Number of components " ,str(int(np.sqrt(total)))


#gmm= GaussianMixture(50) 
gmm = hmm.GaussianHMM(50)
gmm.fit(X_train)


filename1 = 'gmm_avg_face_model.sav'
pickle.dump(gmm, open(filename1, 'wb'))
print("Model saved")

#
#for i in range(0,n-k+1):  #66
#    t=collection_of_frames[i:i+k]
#    x_test = np.array(t).ravel()
#    plt.imshow(np.array(frames[i:i+k]).sum(axis=0) // k)
#    plt.imshow(x_test.reshape(M,N))
#    plt.show()
#    print(i," : ", i+k,"--> ", gmm.score([x_test]))