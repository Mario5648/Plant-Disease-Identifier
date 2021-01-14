import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle
from natsort import natsorted
import pathlib


IMG_SIZE = 200


#this will create the test data that is needed
def create_test_data():
    test_data = []
    i = 0

    for img in natsorted(os.listdir('./Plant_Test')):
        try:
            img_array = cv2.imread(os.path.join('./Plant_Test',img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append(new_array)
        except Exception as e:
            pass

        i += 1
    return test_data

#this can randomize the data order if needed
def randomize_data(data):
    random.shuffle(data)
    return data

test_data = create_test_data()

X=[]

for features in test_data:
    X.append(features)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open("Test.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()



print("[!] Done Processing Data")
