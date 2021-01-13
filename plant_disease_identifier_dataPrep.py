import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle
from natsort import natsorted
import pathlib

#image_id,healthy,multiple_diseases,rust,scab
CATEGORIES = ['Healthy','Multiple_Diseases','Rust','Scab']

IMG_SIZE = 200

#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


#This will identify the category of the picture using the csv file
def match_pic_category():
    training_data_pic_categories = []
    training_data_file = open('./Plant_Data/train.csv', 'r')
    training_data_lines = training_data_file.readlines()

    for line in training_data_lines:
        if line == 'image_id,healthy,multiple_diseases,rust,scab\n':
            continue
        tmpList = list(line.strip('\n').split(','))[1:5]
        training_data_pic_categories.append(CATEGORIES[tmpList.index('1')])

    training_data_file.close()
    return training_data_pic_categories

#this will create the training data that is needed
def create_training_data():
    training_data = []
    i = 0
    training_data_pic_categories = match_pic_category()

    for img in natsorted(os.listdir('./Plant_Train')):
        try:
            img_array = cv2.imread(os.path.join('./Plant_Train',img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, int(CATEGORIES.index(training_data_pic_categories[i]))])
        except Exception as e:
            pass

        i += 1
    return training_data

#this can randomize the data order if needed
def randomize_data(data):
    random.shuffle(data)
    return data
training_data = create_training_data()

X=[]
Y=[]

for features, label in training_data:
    X.append(features)
    Y.append(label)
Y = np.array(Y)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()


print("[!] Done Processing Data")
