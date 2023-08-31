import cv2
import numpy as np
import tensorflow as tf
from vgg_face import verifyFace
from imutils import paths
import os

cc = cv2.VideoCapture(0)
sucess, frame = cc.read()

protxt = 'deploy.prototxt.txt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(protxt, model)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

dataset = './database/'

img_path = list(paths.list_images(dataset))

labels = []


for i in img_path:
    label = i.split(os.path.sep)[-1]
    labels.append(label)
    


while sucess and cv2.waitKey(1) == -1:
    data = []
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)

    detection = net.forward()

    for i in range(0, detection.shape[2]):
        conf = detection[0, 0, i, 2]

        if conf > 0.5:
            box = detection[0, 0, i, 3:7] * np.array([W, H, W, H])
            (sx, sy, ex, ey) = box.astype('int')

            roi = frame[sy:ey, sx:ex]

            cv2.imwrite('img.jpg', roi)
            img = cv2.imread('img.jpg')
            img = cv2.resize(img, (224, 224))
            
            # verify face 
            
            labels = ['./database/gaurav.jpg', './database/prash.jpg']
            
            '''
            result = verifyFace('img.jpg','./database/prash.jpg')
            if result < 0.3:
                cv2.putText(frame,'PRASAHANT',(sx,sy-10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
            
            else:
                cv2.putText(frame, 'UNKNOWN', (sx, sy-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255), 2)
            
            '''
            
            for person in labels:
                result = verifyFace('img.jpg',person)
                if result<0.3:
                    cv2.putText(frame,person,(sx, sy-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    sucess, frame = cc.read()

cv2.destroyAllWindows()
cc.release()
