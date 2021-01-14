from tensorflow import keras
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

CATEGORIES = ['Healthy','Multiple_Diseases','Rust','Scab']


cnn = keras.models.load_model('./Plant_CNN_Model')

T = pickle.load(open("./Test.pickle","rb"))
T=np.array(T/255.0)

y_pred = cnn.predict(T)
y_classes = [np.argmax(element) for element in y_pred]

index = int(input("Enter the Picture Number To See Result: "))
img_array = cv2.imread(os.path.join('./Plant_Test','Test_'+str(index)+'.jpg'))
plt.imshow(img_array)
plt.xlabel(CATEGORIES[y_classes[int(index)]])
plt.show()

print("The plant seems to have/be: "+str(CATEGORIES[y_classes[int(index)]]))
