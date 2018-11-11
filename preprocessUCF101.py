from __future__ import print_function

import os
import numpy as np
import pickle
import datetime
import pdb
import cv2
import pdb
import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def readtxt(filepath,istrain,labeldict = None):
    # print (path)
    videopath = []
    videolabel = []
    with open(filepath, 'r') as f:
        data = f.readlines()
        for line in data:
            if istrain:
                line_split = line.split()
                videopath.append( line_split[0] )
                videolabel.append( int(line_split[1]) )
            else:
                videopath.append( line.split()[0] )
                line_split = line.split("/")
                videolabel.append(int(labeldict[line_split[0]]))
                # pdb.set_trace()
    return videopath,videolabel

def readdict(filepath):
    with open(filepath , 'r') as f:
        dictdata = f.readlines()
        key = []
        value = []
        for line in dictdata:
            line_split = line.split()
            key.append(line_split[1])
            value.append(line_split[0])

    return dict(zip(key,value))

def avi2npy(filelist,labellist,idx,batch_size = 40):
    '''
    Every 3 time points pick 1 time point. And padding the whole sequence into length 50.
    '''
    k = 3
    totallength = 50
    stidx = idx*batch_size
    edidx = min((idx+1)*batch_size,len(labellist))
    label = labellist[stidx:edidx]

    data  = []
    for fidx in range(stidx,edidx):
        filename = "./UCF101/UCF-101/" + filelist[fidx]
        cap = cv2.VideoCapture(filename)
        ret = True
        tempdata = []
        count = 0
        while(ret):
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame',frame)
                if not count % k:
                    frame = cv2.resize(frame,(160,120))
                    tempdata.append(frame)
                count = count + 1
        tempdata = np.asarray(tempdata,dtype=np.float32) / 255.
        # print (tempdata.shape)
        tempdata = pad_sequences([tempdata], maxlen=totallength, truncating='post', dtype='float32')[0]
        data.append(tempdata)
    data = np.asarray(data)
    return data,label

def read_batch_data(partid,batchid,istrain = True):
    folder_path = "./UCF101/processeddata/part"+str(partid)+"/"
    if istrain:
        if os.path.isfile(folder_path+"training/label"+str(batchid)+'.npy'):
            databatch  = np.load(folder_path+"training/data"+str(batchid)+'.npy')
            labelbatch = np.load(folder_path+"training/label"+str(batchid)+'.npy')
        else:
            print ("Run the preprocessUCF101.py first")
    else:
        if os.path.isfile(folder_path+"testing/label"+str(batchid)+'.npy'):
            databatch  = np.load(folder_path+"testing/data"+str(batchid)+'.npy')
            labelbatch = np.load(folder_path+"testing/label"+str(batchid)+'.npy')
        else:
            print ("Run the preprocessUCF101.py first")

    return databatch,labelbatch


if __name__ == '__main__':
    np.random.seed(20160924)
    if not os.path.isdir("./UCF101"):
        os.mkdir("./UCF101")
        print("Add data into ./UCF101 folder, and this folder should have ./UCF101/ucfTrainTestlist and ./UCF101/UCF-101 which can be downloaded from its website")
        exit()

    dictpath = "./UCF101/ucfTrainTestlist/classInd.txt"

    labeldict = readdict(dictpath)

    for idx in [1,2,3]:
        trainfile = "./UCF101/ucfTrainTestlist/trainlist0"+str(idx)+".txt"
        testfile  = "./UCF101/ucfTrainTestlist/testlist0" +str(idx)+".txt"
        trainpath,trainlabel = readtxt(trainfile,istrain= True )
        testpath ,testlabel  = readtxt(testfile ,istrain= False ,labeldict = labeldict)

        if not os.path.isdir("./UCF101/processeddata"):
            os.mkdir("./UCF101/processeddata")

        if not os.path.isdir("./UCF101/processeddata/part"+str(idx)):
            os.mkdir("./UCF101/processeddata/part"+str(idx))

        if not os.path.isdir("./UCF101/processeddata/part"+str(idx)+"/training/"):
            os.mkdir("./UCF101/processeddata/part"+str(idx)+"/training/")

        if not os.path.isdir("./UCF101/processeddata/part"+str(idx)+"/testing/"):
            os.mkdir("./UCF101/processeddata/part"+str(idx)+"/testing/")

        folder_path = "./UCF101/processeddata/part"+str(idx)+"/"

        train_num = len(trainlabel)
        test_num  = len(testlabel)

        shuffle_train = np.random.choice(range(train_num), train_num, False)
        shuffle_test  = np.random.choice(range(test_num ), test_num , False)

        trainpath = np.asarray(trainpath )[shuffle_train]
        trainlabel= np.asarray(trainlabel)[shuffle_train]-1
        trainlabel = to_categorical(trainlabel)

        testpath  = np.asarray(testpath  )[shuffle_test]
        testlabel = np.asarray(testlabel )[shuffle_test]-1
        testlabel = to_categorical(testlabel)

        batch_size = 40

        trainbatchnum = int(train_num/batch_size)+1
        testbatchnum = int(test_num/batch_size)+1

        for bidx in range(trainbatchnum/100):
            if not os.path.isfile(folder_path+"training/label"+str(bidx)+'.npy'):
                databatch,labelbatch = avi2npy(trainpath,trainlabel,bidx,batch_size = 40)
                np.save(folder_path+"training/data"+str(bidx)+'.npy',databatch)
                np.save(folder_path+"training/label"+str(bidx)+'.npy',labelbatch)
            print ("Finished training batch number : "+str(bidx))

        for bidx in range(testbatchnum/30):
            if not os.path.isfile(folder_path+"testing/label"+str(bidx)+'.npy'):
                databatch,labelbatch = avi2npy(testpath,testlabel,bidx,batch_size = 40)
                np.save(folder_path+"testing/data"+str(bidx)+'.npy',databatch)
                np.save(folder_path+"testing/label"+str(bidx)+'.npy',labelbatch)
            print ("Finished testing batch number : "+str(bidx))





