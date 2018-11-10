import os
import numpy as np
import pickle
import datetime
import pdb
import cv2
import pdb
import random

from keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    if not os.path.isdir("./UCF101"):
        os.mkdir("./UCF101")
        print("Add data into ./UCF101 folder, and this folder should have ./UCF101/ucfTrainTestlist and ./UCF101/UCF-101 which can be downloaded from its website")