import cv2
import sys
import numpy

# Get user supplied values
imagePath = "1.jpg"
faceCascPath = "haarcascade_frontalface_default.xml"
eyeCascPath = "haarcascade_eye.xml"
mouthCascPath = "haarcascade_mcs_mouth.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(faceCascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)
mouthCascade = cv2.CascadeClassifier(mouthCascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(roi_gray, 1.1)
    i = 0
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        i = i+1
        if (i == 2) :
            break
    mouth = mouthCascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=7)
    for (mx,my,mw,mh) in mouth:
        my = int(my - 0.15*mh)
        coord = [mx,my,mx+mw, my+mh]
        cv2.rectangle(roi_color, (mx,my), (mx+mw,my+mh), (0,0,255), 2)
        #break
    cv2.rectangle(roi_color, (coord[0],coord[1]), (coord[2],coord[3]), (255,0,255), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)