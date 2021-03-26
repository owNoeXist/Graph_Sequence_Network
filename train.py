import os
import numpy as np
from datetime import datetime

#========================================================================
def TrainModel(MODEL, TRAIN_DATA, TRAIN_TEST, VALID_TEST, MODEL_PATH, TRAIN_EPOCH, TEST_FREQ, SAVE_FREQ, RESULT_PATH):
    result = {"TrainStep":[],"TrainLoss":[],"ValidLoss":[],"TestStep":[],"TrainAuc":[],"ValidAuc":[]}
    #Model initial state
    trainLoss,trainAuc = MODEL.TestModel(TRAIN_TEST)
    validLoss,validAuc = MODEL.TestModel(VALID_TEST)
    result["TrainStep"].append(0)
    result["TrainLoss"].append(trainLoss)
    result["ValidLoss"].append(validLoss)
    result["TestStep"].append(0)
    result["TrainAuc"].append(trainAuc)
    result["ValidAuc"].append(validAuc)
    MODEL.say("Initial state = {}".format(result))
    #Train by EPOCH
    bestAuc=0
    for i in range(1,TRAIN_EPOCH+1):
        MODEL.say("EPOCH {0}/{1} @ {2}".format(i, TRAIN_EPOCH, datetime.now()))
        #train for one epoch
        trainLoss = MODEL.TrainModel(TRAIN_DATA)
        validLoss,validAuc = MODEL.TestModel(VALID_TEST)
        result["TrainStep"].append(i)
        result["TrainLoss"].append(trainLoss)
        result["ValidLoss"].append(validLoss)
        MODEL.say("Training Loss= {0}\nValidate Loss= {1}".format(trainLoss,validLoss))
        #test model
        if i % TEST_FREQ == 0:  
            _,trainAuc = MODEL.TestModel(TRAIN_TEST)
            result["TestStep"].append(i)
            result["TrainAuc"].append(trainAuc)
            result["ValidAuc"].append(validAuc)
            MODEL.say("Training Auc= {0}\nValidate Auc= {1}".format(trainAuc,validAuc))
            if validAuc > bestAuc :
                bestAuc = validAuc
                MODEL.save(MODEL_PATH, NAME="_best")
        if i % SAVE_FREQ == 0:
            MODEL.save(MODEL_PATH)
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="TrainLossAuc")

#========================================================================
def GetTopSim(MODEL, DATAS, RESULT_PATH, SELECT=0, K=0):
    result = {"SerialNum":[]}
    all = len(DATAS.rawData[SELECT*2])
    for i in range(all):
        data = DATAS.GetTopkEpoch(DATAS.rawData[SELECT*2][i], DATAS.rawData[SELECT*2+1])
        label = MODEL.GetTopkLabel(data)
        betterNum = 1
        for j in range(all):
            if label[j] > label[i]:
                betterNum += 1
        result["SerialNum"].append(betterNum)
        print("{0}/{1} {2}".format(i, all, datetime.now()))
    if K!=0:
        num = 0
        for i in range(all):
            if result["SerialNum"][i] <= K:
                num += 1
        print("sum/all:{0}/{1}".format(num,all))
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="TopSerial")

#========================================================================
def MatchFunction(MODEL, DATAS, RESULT_PATH):
    result = {"auc":0, "match":[],"funcName":[]}
    #Get all cos
    num = 0
    allCos = []
    funcName = []
    while True:
        data = DATAS.GetTopkEpoch(num,DATAS.rawData[4][num],DATAS.rawData[5])
        if data == []:
            break
        cos = MODEL.GetTopkCos(data)
        allCos.append(cos)
        funcName.append("{0}.{1}".format(num,DATAS.rawData[4][num].name))
        num+=1
        print("{0}/{1} {2}".format(num,len(cos),datetime.now()))
    #Match function
    match = []
    allSort = []
    distance = []
    for value in allCos:
        sort = cosSort(value)
        allSort.append(sort)
        distance.append(sort[0][0]-sort[1][0])
    for i in range(len(distance)):
        max = maxDistance(distance)
        for j in range(len(allSort[max])):
            if repeatCheck(allSort[max][j][1],match):
                match.append((max,allSort[max][j][1]))
                break
        distance[max]=0
        updateDistance(match,allSort,distance)
    matched=0
    for value in match:
        if value[0]==value[1]:
            matched+=1
    print(match)
    print(funcName)
    print("{0}/{1}".format(matched,len(match)))
    #Save result
    result["auc"] = matched/len(match)
    result["match"] = match
    result["funcName"] = funcName
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="match")

def cosSort(cos):
    sort = {}
    for i in range(len(cos)):
        sort[cos[i].numpy()]=i
    return sorted(sort.items(),reverse=True)

def maxDistance(DISTANCE):
    max=0
    maxDst=0
    for i in range(len(DISTANCE)):
        if DISTANCE[i] > maxDst:
            max=i
            maxDst=DISTANCE[i]
    return max

def repeatCheck(value,MATCH):
    for i in range(len(MATCH)):
        if value==MATCH[i][1]:
            return False
    return True

def updateDistance(MATCH,ALLSORT,DISTANCE):
    for i in range(len(DISTANCE)):
        if DISTANCE[i]!=0:
            j=0
            for j in range(len(ALLSORT[i])):
                if repeatCheck(ALLSORT[i][j][1],MATCH):
                    break
            k=0
            for k in range(j+1,len(ALLSORT[i])):
                if repeatCheck(ALLSORT[i][k][1],MATCH):
                    break
            DISTANCE[i]=ALLSORT[i][j][0]-ALLSORT[i][k][0]
   
    