import cv2
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import os
import time

cc = cv2.VideoCapture(0)
sucess, frame = cc.read()

protxt = '../deploy.prototxt.txt'
model = '../res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(protxt, model)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

dataset = './database/'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--person_name', dest='person_name',metavar='person_name', required=True, help='PERSON NAME')
    return parser



parser = build_parser()
args = parser.parse_args()

count = 0

timer = int(10)

prev = time.time()

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
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            
            
            if not os.path.exists('./database/'+args.person_name):
                os.makedirs('./database/'+args.person_name)
            
            cur = time.time()
            if count<=100:
                if count%10==0:
                    cv2.imwrite('./database/'+args.person_name +'/'+str(count)+'.jpg', roi)
                
            else:
                break
                
            count+=1
                    
                

    cv2.imshow('frame', frame)
    sucess, frame = cc.read()

cv2.destroyAllWindows()
cc.release()
