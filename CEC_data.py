import os
import json
import numpy as np

import cfg

class MyData():
    def __init__(self, HYPER_PARAMETER, DATA_PATH, PARTITIONS):
        self.Line1st = HYPER_PARAMETER['Line1st']
        self.Word1st = HYPER_PARAMETER['Word1st']
        self.Line2nd = HYPER_PARAMETER['Line2nd']
        self.Word2nd = HYPER_PARAMETER['Word2nd']
        self.BatchSize = HYPER_PARAMETER['BatchSize']
        self.rawData = self.ObtainRawData(DATA_PATH,PARTITIONS)
    
    def GetTrainData(self):
        return self.GetEpoch(self.rawData[0],self.rawData[1])

    def GetValidData(self):
        return self.GetEpoch(self.rawData[2],self.rawData[3])

    def GetTestData(self):
        return self.GetEpoch(self.rawData[4],self.rawData[5])

    def ObtainRawData(self, DATA_PATH, PARTITIONS):
        #Obtain data from each file folder
        data = [[],[],[],[],[]]
        graphInfo = {}
        for dir in os.listdir(DATA_PATH):
            curFile=0
            programPath=os.path.join(DATA_PATH,dir)
            if not os.path.isdir(programPath):
                continue
            for file in os.listdir(programPath):
                filePath=os.path.join(programPath,file)
                with open(filePath) as lines:
                    for line in lines:
                        graphInfo = json.loads(line.strip())
                        graphInfo['Funcname']=dir+'-'+graphInfo['Funcname']
                        curCFG = cfg.CFG(graphInfo['Funcname'],graphInfo['Nodenum'])
                        for u in range(graphInfo['Nodenum']):
                            curCFG.basic[u] =np.array(graphInfo['NodeWords'][u],dtype=object)
                            for v in graphInfo['NodeTo'][u]:
                                curCFG.AddDirectedEdge(u, v)
                        data[curFile].append(curCFG)
                curFile+=1
        print("{0} C functions, {1} X86-O1 functions, {2} X86-O2 functions, {3} X86-O3 functions,\
            {4} X86-O4 functions".format(len(data[0]),len(data[1]),len(data[2]),len(data[3]),len(data[4])))

        #Match function
        dataX = []
        dataY = []
        for graphX in data[0]:
            for i in range(1,5):
                for graphY in data[i]:
                    if graphX.name == graphY.name and graphX.nodeNum < 50 \
                        and 1/2 < graphY.nodeNum/graphX.nodeNum < 2:
                        dataX.append(graphX)
                        dataY.append(graphY)
                        data[i].remove(graphY)
                        break
        print("{0} matched functions".format(len(dataX)))

        #Part Data
        start = 0.0
        dataPart = []
        funcNum = len(dataX)
        permPath=os.path.join(DATA_PATH,"perm.npy")
        if os.path.isfile(permPath):
            perm = np.load(permPath)
        else:
            perm = np.random.permutation(len(dataX))
            np.save(permPath, perm)
        if len(perm) != len(dataX):
            perm = np.random.permutation(len(dataX))
            np.save(permPath, perm)
        for part in PARTITIONS:
            curDataX = [] 
            curDataY = [] 
            end = start + part * funcNum
            for cls in range(int(start), int(end)):
                curDataX.append(dataX[perm[cls]])
                curDataY.append(dataY[perm[cls]])
            dataPart.append(curDataX)
            dataPart.append(curDataY)
            start = end

        return dataPart

    def GetEpoch(self, DATA_X, DATA_Y):
        start = 0
        epochData = []
        funcNum = len(DATA_X)
        while start < funcNum:
            if start + self.BatchSize >= funcNum:
                break
            x1, x2, m1, m2, y = self.GetPair(DATA_X, DATA_Y, START=start)
            epochData.append((x1, x2, m1, m2, y))
            start += self.BatchSize
        return epochData

    def GetPair(self, DATA_X, DATA_Y, START = -1, VALUE = 2):
        pairs = []
        maxN1 = 0
        maxN2 = 0 
        funcNum = len(DATA_X)
        end =min(START + self.BatchSize, funcNum)
        posPairNum = end-START
        allPairNum = VALUE*posPairNum

        for cls1 in range(START, end):
            #build positive pair
            pairs.append((cls1,cls1))
            maxN1 = max(maxN1, DATA_X[cls1].nodeNum)
            maxN2 = max(maxN2, DATA_Y[cls1].nodeNum)
            #build negetive pair
            for i in range(VALUE-1):
                cls2 = np.random.randint(funcNum)
                while (cls2 == cls1):
                    cls2 = np.random.randint(funcNum)
                pairs.append((cls1,cls2))
                maxN2 = max(maxN2, DATA_Y[cls2].nodeNum)

        xInput1 = np.zeros((allPairNum, maxN1, self.Line1st, self.Word1st))
        nodeMask1 = np.zeros((allPairNum, maxN1, maxN1))
        xInput2 = np.zeros((allPairNum, maxN2, self.Line2nd, self.Word2nd))
        nodeMask2 = np.zeros((allPairNum, maxN2, maxN2))
        yInput = np.zeros((allPairNum))

        for i in range(allPairNum):
            graphX = DATA_X[pairs[i][0]]
            for u in range(graphX.nodeNum):
                l=0
                for line in graphX.basic[u]:
                    if l>=self.Line1st:
                        break
                    w=0
                    for word in line:
                        if w>=self.Word1st:
                            break
                        xInput1[i, u, l, w] = word
                        w+=1
                    l+=1
                for v in graphX.succs[u]:
                    nodeMask1[i, u, v] = 1
            graphY = DATA_Y[pairs[i][1]]
            for u in range(graphY.nodeNum):
                l=0
                for line in graphY.basic[u]:
                    if l>=self.Line2nd:
                        break
                    w=0
                    for word in line:
                        if w>=self.Word2nd:
                            break
                        xInput2[i, u, l, w] = word
                        w+=1
                    l+=1
                for v in graphY.succs[u]:
                    nodeMask2[i, u, v] = 1
            if i%VALUE == 0:
                yInput[i] = 1
            else:
                yInput[i] = 0

        return xInput1,xInput2,nodeMask1,nodeMask2,yInput
