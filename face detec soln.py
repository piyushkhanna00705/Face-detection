# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 23:46:18 2019

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

nadia=cv2.imread('Nadia_Murad.jpg',0)
denis=cv2.imread('Denis_Mukwege.jpg',0)
solvay=cv2.imread('solvay_conference.jpg',0)

plt.imshow(nadia,cmap='gray')
plt.imshow(denis,cmap='gray')

face_cascade=cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img=img.copy()
    face_rects=face_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

result=detect_face(denis)
plt.imshow(result,cmap='gray')

result=detect_face(nadia)
plt.imshow(result,cmap='gray')

result=detect_face(solvay)
plt.imshow(result,cmap='gray')

#adjusting haar cascade detector

def adj_detect_face(img):
    face_img=img.copy()
    face_rects=face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

result=adj_detect_face(solvay)
plt.imshow(result,cmap='gray')

#Eye Detection

eye_cascade=cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    
    face_img=img.copy()
    eyes=eye_cascade.detectMultiScale(face_img)
    
    for (x,y,w,h) in eyes:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

result=detect_eyes(nadia)
plt.imshow(result,cmap='gray')

eyes=eye_cascade.detectMultiScale(denis)

#White arounf pupils not distict in denis image..

result=detect_eyes(denis)
plt.imshow(result,cmap='gray')

    
