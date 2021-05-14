import os
import numpy as np
from tqdm import tqdm
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
    bestAuc = validAuc
    for i in range(1,TRAIN_EPOCH+1):
        MODEL.say("Train Epoch {0}/{1} @ {2}".format(i, TRAIN_EPOCH, datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
        #train for one epoch
        trainLoss = MODEL.TrainModel(TRAIN_DATA)
        validLoss,validAuc = MODEL.TestModel(VALID_TEST)
        result["TrainStep"].append(i)
        result["TrainLoss"].append(trainLoss)
        result["ValidLoss"].append(validLoss)
        MODEL.say("Training Loss= {0:.6f}\nValidate Loss= {1:.6f}".format(trainLoss,validLoss))
        #test model
        if i % TEST_FREQ == 0:  
            _,trainAuc = MODEL.TestModel(TRAIN_TEST)
            result["TestStep"].append(i)
            result["TrainAuc"].append(trainAuc)
            result["ValidAuc"].append(validAuc)
            MODEL.say("Training Auc= {0:.6f}\nValidate Auc= {1:.6f}".format(trainAuc,validAuc))
            if validAuc > bestAuc :
                bestAuc = validAuc
                MODEL.save(MODEL_PATH, NAME="_best")
        if i % SAVE_FREQ == 0:
            MODEL.save(MODEL_PATH)
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="TrainLossAuc")

#========================================================================
def GetRoc(MODEL, RESULT_PATH, TRAIN_DATAS=None, VALID_DATAS=None):
    result = {}
    if TRAIN_DATAS!=None:
        result['TrainFpr'],result['TrainTpr'],result['TrainAuc'] = MODEL.TestModel(TRAIN_DATAS,SAVE=True)
    if VALID_DATAS!=None:
        result['ValidFpr'],result['ValidTpr'],result['ValidAuc'] = MODEL.TestModel(VALID_DATAS,SAVE=True)
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="RocAuc")

#========================================================================
def GetTopSim(MODEL, DATAS, RESULT_PATH, SELECT=0, K=0):
    result = {"SerialNum":[],"FuncName":[],"TopSimName":[]}
    funcNum = len(DATAS.rawData[SELECT*2])
    for i in tqdm(range(funcNum),desc='GetTopSim',unit='func'):
        data = DATAS.GetOneFuncEpoch(DATAS.rawData[SELECT*2][i], DATAS.rawData[SELECT*2+1])
        label = MODEL.GetLabel(data)
        betterNum = 1
        bestNum = 0
        bestSim = label[0]
        for j in range(funcNum):
            if label[j] > label[i]:
                betterNum += 1
            if label[j] > bestSim:
                bestNum = j
                bestSim = label[j]
        result["SerialNum"].append(betterNum)
        result["FuncName"].append(DATAS.rawData[SELECT*2][i].name)
        result["TopSimName"].append(DATAS.rawData[SELECT*2][bestNum].name)
    if K!=0:
        num = 0
        for i in range(funcNum):
            if result["SerialNum"][i] <= K:
                num += 1
        result["Top{}".format(K)] = num/funcNum
        print("Top{0} similarity(sum/all):{1}/{2}  {3:.6f}".format(K,num,funcNum,num/funcNum))
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="TopSerial")

#========================================================================
def MatchFunction(MODEL, DATAS, RESULT_PATH, SELECT=0):
    result = {"MatchNum":[],"FuncName":[]}
    #Get all label
    allLabel = []
    funcNum = len(DATAS.rawData[SELECT*2])
    for i in tqdm(range(funcNum),desc='MatchFunction',unit='func'):
        data = DATAS.GetOneFuncEpoch(DATAS.rawData[SELECT*2][i],DATAS.rawData[SELECT*2+1])
        label = MODEL.GetLabel(data)
        allLabel.append(label)
        result["FuncName"].append(DATAS.rawData[SELECT*2][i].name)
    result["MatchNum"] = matchFunc(allLabel)
    #Calculate auc
    matched=0
    for value in result["MatchNum"]:
        if value[0]==value[1]:
            matched+=1
    result["Auc"] = matched/funcNum
    print("{0}/{1}".format(matched,funcNum))
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="Match")

def matchFunc(ALL_LABEL):
    funcNum = len(ALL_LABEL)
    #initial
    allSort = []
    distance = []
    for label in ALL_LABEL:
        sortList = labelSort(label)
        allSort.append(sortList)
        distance.append([sortList[0][0]-sortList[1][0],sortList[0][1],sortList[1][1]])
    #match
    match = []
    matched = [0]*funcNum
    for i in range(funcNum):
        maxDis = maxDistance(distance)
        distance[maxDis][0] = 0
        match.append((maxDis,distance[maxDis][1]))
        matched[distance[maxDis][1]]=1
        updateDistance(matched,allSort,distance)
    return match

def labelSort(label):
    sortDic = {}
    for i in range(len(label)):
        sortDic[label[i].numpy()] = i
    return sorted(sortDic.items(),reverse=True)

def maxDistance(DISTANCE):
    max = 0
    maxDst = 0
    for i in range(len(DISTANCE)):
        if DISTANCE[i][0] > maxDst:
            max = i
            maxDst = DISTANCE[i][0]
    return max

def updateDistance(MATCHED,ALLSORT,DISTANCE):
    for i in range(len(DISTANCE)):
        if DISTANCE[i][0] != 0 and (MATCHED[DISTANCE[i][1]] == 1 or MATCHED[DISTANCE[i][2]] == 1):
            frist = 0
            for frist in range(len(ALLSORT[i])):
                if MATCHED[ALLSORT[i][frist][1]] == 0:
                    break
            second = 0
            for second in range(frist+1,len(ALLSORT[i])):
                if MATCHED[ALLSORT[i][second][1]] == 0:
                    break
            DISTANCE[i]=[ALLSORT[i][frist][0]-ALLSORT[i][second][0],ALLSORT[i][frist][1],ALLSORT[i][second][1]]


def MatchFunctionTest(MODEL, DATAS, RESULT_PATH, SELECT=0):
    result = {"MatchNum":[],"FuncName":[]}
    #Get all label
    allLabel = []
    funcNum = len(DATAS.rawData[SELECT*2])
    for i in tqdm(range(funcNum),desc='MatchFunction',unit='func'):
        data = DATAS.GetOneFuncEpoch(DATAS.rawData[SELECT*2][i],DATAS.rawData[SELECT*2+1])
        label = MODEL.GetLabel(data)
        allLabel.append(label)
        result["FuncName"].append(DATAS.rawData[SELECT*2][i].name)
    result["MatchNum"] = matchFuncTest(allLabel)
    '''
    #Calculate auc
    matched=0
    for value in result["MatchNum"]:
        if value[0]==value[1]:
            matched+=1
    result["Auc"] = matched/funcNum
    print("{0}/{1}".format(matched,funcNum))
    MODEL.saveResult(result, RESULT_PATH, FILE_NAME="Match")
    '''

def matchFuncTest(ALL_LABEL):
    funcNum = len(ALL_LABEL)

    numSet = []
    valueSet = []
    allSort = []
    for i in range(funcNum):
        sortList = labelSort(ALL_LABEL[i])
        allSort.append(sortList)
        
        mergeSet = []
        setNum = len(valueSet)
        for i in range(setNum):
            for j in range(3):
                if sortList[j][1] in valueSet[i]:
                    mergeSet.append(i)
                    break

        j = len(mergeSet)-1
        while j > 0:
            numSet[mergeSet[0]].extend(numSet[mergeSet[j]])
            numSet=numSet[:mergeSet[j]]+numSet[mergeSet[j]+1:]
            valueSet[mergeSet[0]].extend(valueSet[mergeSet[j]])
            valueSet=valueSet[:mergeSet[j]]+valueSet[mergeSet[j]+1:]
            mergeSet = mergeSet[:j]
            j-=1

        insertIndex = -1
        if len(mergeSet) == 0:
            numSet.append([])
            valueSet.append([])
        else:
            insertIndex = mergeSet[0]
        numSet[insertIndex].append(i)
        for k in range(3):
            if sortList[k][1] not in valueSet[insertIndex]:
                valueSet[insertIndex].append(sortList[k][1])
    print(numSet)
    print(valueSet)
