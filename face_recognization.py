from numpy.core.numeric import empty_like
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, LocallyConnected2D, Flatten, Dense, Dropout, Dense
from tensorflow.python.keras.layers.core import Activation
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from imutils import paths
from tqdm import tqdm
import os
import cv2
from datetime import date
#from vgg_face import preprocess

base_model = Sequential()
base_model.add(Convolution2D(32, (11, 11), activation='relu',
                             name='C1', input_shape=(152, 152, 3)))
base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
base_model.add(LocallyConnected2D(
    16, (7, 7), strides=2, activation='relu', name='L5'))
base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
base_model.add(Flatten(name='F0'))
base_model.add(Dense(4096, activation='relu', name='F7'))
base_model.add(Dropout(rate=0.5, name='D0'))
base_model.add(Dense(8631, activation='softmax', name='F8'))

base_model.load_weights('VGGFace2_DeepFace_weights_val-0.9034.h5')

deepface_model = Model(
    base_model.layers[0].input, base_model.layers[-3].output)


def preprocess(img):
    img = load_img(img, target_size=(152, 152))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    return img


def l2_normalize(x):
    return x/np.sqrt(np.sum(np.multiply(x, x)))


def findEuclideanDistance(source_representaion, test_representation):
    euclidean_distance = source_representaion - test_representation
    euclidean_distance = np.sum(np.multiply(
        euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#img1_embedding = deepface_model.predict(preprocess('img.jpg'))[0]
#img2_embedding = deepface_model.predict(preprocess('./database/prash.jpg'))[0]

#euclidean_distance = findEuclideanDistance(l2_normalize(img1_embedding),l2_normalize(img2_embedding))


dataset = './database/'

img_paths = list(paths.list_images(dataset))

employees = dict()

for i in img_paths:
    employee = i.split(os.path.sep)[-2]
    #img_path = 'database/%s.jpg' % (employee)
    img = preprocess(i)

    representation = deepface_model.predict(img)

    employees[employee] = representation


# -----------------------------------------------------------------------------------------
# face-reco


class face_recognize:
    def write_to_file(self,text):
        with open('attendence.txt', 'a+') as fs:
            fs.seek(0)
            data = fs.read(100)
            if len(data) > 0:
                fs.write('\n')
            fs.write(text+' '+str(date.today())+' '+'PRESENT')
            cv2.destroyAllWindows()
            self.cc.release()
 
    def face(self):
        self.cc = cv2.VideoCapture(0)
        sucess, frame = self.cc.read()
    
        protxt = './deploy.prototxt.txt'
        model = './res10_300x300_ssd_iter_140000.caffemodel'
    
        net = cv2.dnn.readNetFromCaffe(protxt, model)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

        while sucess and cv2.waitKey(1) == -1:
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

                    cv2.imwrite('img.jpg', roi)

                    captured_representation = deepface_model.predict(preprocess('img.jpg'))[0]

                    distances = []

                    for i in employees:
                        employee_name = i
                        #print(employee_name)
                        source_representation = employees[i]

                        distance = findEuclideanDistance(l2_normalize(captured_representation), l2_normalize(source_representation))
                        distances.append(distance)

                    #print(np.argmin(distance))

                    is_found = False
                    idx = 0
                    for i in employees:
                        employee_name = i
                        #print(employee_name)

                        if idx == np.argmin(distances):
                            if distances[idx] <= 0.70:
                                print('detected :', employee_name,'(', distances[idx], ')')
                                employee_name = employee_name.replace('_', '')
                                similarity = distances[idx]
                                is_found = True
                                #print(employee_name)
                                break

                        idx = idx+1
                    if is_found:
                        #label = employee_name+'('+'{0:.2f}'.format(similarity)+')'
                        employee_name = employee_name.replace('./database/', '')
                        label = employee_name.upper()

                        self.write_to_file(label)

                        cv2.putText(frame, str(label), (sx, sy-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    
                    cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            sucess, frame = self.cc.read()

        cv2.destroyAllWindows()
        self.cc.release()

