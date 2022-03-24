import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
import numpy as np
#import imutils
import time
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


cap = cv2.VideoCapture(0)

def detect_and_predict_mask(frame, maskNet):

    locs=[]
    faces = [] 
    preds=[]


    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=10)

    for (x, y, w, h) in detections:
        roi = frame[y:y+h, x:x+w]
        locs.append((x,y,w,h))
        face = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)

    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    
    return (locs,preds)

while True:
    ret, frame=cap.read()

    maskNet = load_model("mask_detector.model")

    (locs,preds) = detect_and_predict_mask(frame, maskNet)

    for (loc, pred) in zip(locs, preds):

        (x,y,w,h)=loc

        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0) , 2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()