import os
import sys
from shutil import copyfile


for filename in os.listdir("./plant_Data/images"):
    if str(filename)[0:5] == "Train":
        copyfile("./plant_Data/images/"+str(filename) , "./Plant_Train/"+str(filename) )
