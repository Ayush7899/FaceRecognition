import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    nimg = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        
        scaleFactor=1.2,
        minNeighbors=5
        ,     
        minSize=(20, 20)
    )
    abcd_image=img[0:img.shape[0], 0:img.shape[1]]
    
    blur_image = cv2.GaussianBlur(abcd_image, (21, 21), 10)
    
    for (x,y,w,h) in faces:
        img[0:img.shape[0], 0:img.shape[1]]=blur_image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img[y:y+h, x:x+w]=nimg[y:y+h, x:x+w]
    

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
