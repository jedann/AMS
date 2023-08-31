import cv2
import numpy as np
import tensorflow as tf
import os
import time

class face_detect:

    
    def face_detected(person_name):
        cc = cv2.VideoCapture(0)
        sucess, frame = cc.read()
        protxt = './deploy.prototxt.txt'
        model = './res10_300x300_ssd_iter_140000.caffemodel'
    
        net = cv2.dnn.readNetFromCaffe(protxt, model)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        dataset = './database/'
   
        while sucess:
            
            cv2.putText(frame,'s == SAVE DETECTED IMAGE',(40,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
            cv2.putText(frame,'p == PREVIEW IMAGE',(40,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
            cv2.putText(frame, 'q == EXIT', (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            

            
            k = cv2.waitKey(3)
            
        
            
            data = []
            (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
            net.setInput(blob)
            detection = net.forward()

            for i in range(0, detection.shape[2]):
                conf = detection[0, 0, i, 2]
            
                if conf > 0.5:
                    box = detection[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (sx, sy, ex, ey) = box.astype('int')
                
                    roi = frame[sy:ey, sx:ex]
                    
                    # save roi to database 
                    
                    #if not os.path.exists('./database/'+person_name):
                    #    os.makedirs('./database/'+person_name)
                    
                    if k==ord('s'):
                        path = './database/'+person_name+'/'+'1.jpg'
                        cv2.imwrite(path,roi)
                        
                    if k == ord('p'):
                        img1 = cv2.imread('./database/'+person_name+'/'+'1.jpg')
                        cv2.imshow('preview',img1)
                        cv2.waitKey()
                        


                    cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            
            cv2.imshow('frame', frame)
            sucess, frame = cc.read()
            
            if k == ord('q'):
                cv2.destroyAllWindows()
                cc.release()
        
        

        cv2.destroyAllWindows()
        cc.release()


