
from random import randrange
import cv2

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img = cv2.imread('download.jfif')
web = cv2.VideoCapture(0)
while True:
    frameread , img = web.read()

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    facecordinates = trained_data.detectMultiScale(grayscale)
    for(x,y,w,h) in facecordinates:
        cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(255),randrange(255)),4)
    cv2.imshow('Hello',img)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break
web.release()


