import os
import json
import numpy as np

from PCFG import PCFG

class MyData():
    def __init__(self, HYPER_PARAMETER, DATA_PATH, PARTITIONS):
        self.Literal1st = HYPER_PARAMETER['Literal1st']
        self.Semantic1st = HYPER_PARAMETER['Semantic1st']
        self.Literal2nd = HYPER_PARAMETER['Literal2nd']
        self.Semantic2nd = HYPER_PARAMETER['Semantic2nd']
        self.BatchSize = HYPER_PARAMETER['BatchSize']
        self.BatchMultiple = HYPER_PARAMETER['BatchMultiple']
        self.rawData = self.ObtainRawData(DATA_PATH,PARTITIONS)

    def ObtainRawData(self, DATA_PATH, PARTITIONS):
        #Go through software folder and exact function into class
        data1st = []
        data2nd = []
        for dir in os.listdir(DATA_PATH):
            programPath = os.path.join(DATA_PATH,dir)
            if not os.path.isdir(programPath):
                continue
            #Go through file 
            for file in os.listdir(programPath):
                filePath = os.path.join(programPath,file)
                data = []
                with open(filePath) as lines:
                    #Exact function PCFG
                    for line in lines:
                        graphInfo = json.loads(line.strip())
                        curPCFG = PCFG(dir+'-'+graphInfo['Funcname'], graphInfo['Nodenum'])
                        for u in range(graphInfo['Nodenum']):
                            for v in graphInfo['CFG'][u]:
                                curPCFG.AddDirectedEdgeCFG(u, v)
                            curPCFG.literal[u]  = np.array(graphInfo['Literal'][u], dtype=object)
                            curPCFG.semantic[u] = np.array(graphInfo['Semantic'][u], dtype=object)
                        data.append(curPCFG)
                #Sort function by name and delete repeat function
                data.sort(key=lambda PCFG:PCFG.name)
                i=0
                while i<len(data)-1:
                    if data[i].name == data[i+1].name:
                        if data[i].IsSame(data[i+1]):
                            data[i:]=data[i+1:]
                        else:    
                            j=2
                            while i+j < len(data) and data[i].name == data[i+j].name:
                                j+=1
                            data[i:]=data[i+j:]
                    else:
                        i+=1
                if file not in ['DataC.json']:
                    data1st.append(data)
                else:
                    data2nd.append(data)
            print("{0} X86 functions, {1} C functions".format(len(data1st[-1]),len(data2nd[-1])))
        
        #match function
        for i in range(len(data1st)):
            j=0
            while j < len(data1st[i]) and j < len(data2nd[i]):
                if data1st[i][j].name==data2nd[i][j].name:
                    if data2nd[i][j].nodeNum <= 50 and 0.5 <= data1st[i][j].nodeNum/data2nd[i][j].nodeNum <= 2:
                        j+=1
                    else:
                        data1st[i][j:]=data1st[i][j+1:]
                        data2nd[i][j:]=data2nd[i][j+1:]
                elif data1st[i][j].name < data2nd[i][j].name:
                    data1st[i][j:]=data1st[i][j+1:]
                else:
                    data2nd[i][j:]=data2nd[i][j+1:]
            if j < len(data1st[i]):
                data1st[i][j:]=[]
            if j < len(data2nd[i]):
                data2nd[i][j:]=[]
            print("{0} matched functions".format(len(data1st[i])))

        #Load perm file to use before data part
        dataPart1st = []
        dataPart2nd = []
        for i in range(len(data1st)):
            dataPart1st.extend(data1st[i])
            dataPart2nd.extend(data2nd[i])
        funcNum = len(dataPart1st)
        permPath = os.path.join(DATA_PATH,"perm.npy")
        if os.path.isfile(permPath):
            perm = np.load(permPath)
        else:
            perm = np.random.permutation(funcNum)
            np.save(permPath, perm)
        if len(perm) != funcNum:
            perm = np.random.permutation(funcNum)
            np.save(permPath, perm)
        
        #Part Data
        start = 0.0
        dataPart = []
        for part in PARTITIONS:
            partData1st = [] 
            partData2nd = [] 
            end = start + part * funcNum
            for cls in range(int(start), int(end)):
                partData1st.append(dataPart1st[perm[cls]])
                partData2nd.append(dataPart2nd[perm[cls]])
            dataPart.append(partData1st)
            dataPart.append(partData2nd)
            start = end

        return dataPart

    def GetTrainData(self,SELECT=0):
        return self.GetTrainEpoch(self.rawData[SELECT*2],self.rawData[SELECT*2+1])

    def GetTrainEpoch(self, DATA1ST, DATA2ND):
        start = 0
        pairs = []
        epochData = []
        funcNum = len(DATA1ST)
        while start < funcNum:
            c1, p1, l1 ,s1 ,c2, p2, l2, s2, y, pair = self.GetTrainPair(DATA1ST, DATA2ND, START=start)
            epochData.append((c1, p1, l1 ,s1 ,c2, p2, l2, s2, y))
            pairs += pair
            start += self.BatchSize
        return epochData,pairs

    def GetTrainPair(self, DATA1ST, DATA2ND, START = -1):
        pairs = []
        maxNode1st = 0
        maxNode2nd = 0 
        funcNum = len(DATA1ST)
        end =min(START + self.BatchSize, funcNum)
        posPairNum = end-START
        allPairNum = self.BatchMultiple*posPairNum

        for cls1 in range(START, end):
            #build positive pair
            pairs.append((cls1,cls1))
            maxNode1st = max(maxNode1st, DATA1ST[cls1].nodeNum)
            maxNode2nd = max(maxNode2nd, DATA2ND[cls1].nodeNum)
            #build negetive pair
            for i in range(self.BatchMultiple-1):
                cls2 = np.random.randint(funcNum)
                while cls2 == cls1:
                    cls2 = np.random.randint(funcNum)
                pairs.append((cls1,cls2))
                maxNode2nd = max(maxNode2nd, DATA2ND[cls2].nodeNum)

        cfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        pfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        pfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        yInput = np.zeros((allPairNum))

        for i in range(allPairNum):
            graph1st = DATA1ST[pairs[i][0]]
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsPFG[u]:
                    pfg1st[i, u, v] = 1
                for l in range(len(graph1st.literal[u])):
                    literal1st[i, u, l] = graph1st.literal[u][l]
                for w in range(len(graph1st.semantic[u])):
                    if w >= self.Semantic1st:
                        break
                    semantic1st[i, u, w] = graph1st.semantic[u][w]
                
            graph2nd = DATA2ND[pairs[i][1]]
            for u in range(graph2nd.nodeNum):
                for v in graph2nd.succsCFG[u]:
                    cfg2nd[i, u, v] = 1
                for v in graph2nd.succsPFG[u]:
                    pfg2nd[i, u, v] = 1
                for l in range(len(graph2nd.literal[u])):
                    literal2nd[i, u, l] = graph2nd.literal[u][l]
                for w in range(len(graph2nd.semantic[u])):
                    if w >= self.Semantic2nd:
                        break
                    semantic2nd[i, u, w] = graph2nd.semantic[u][w]

            if i%self.BatchMultiple == 0:
                yInput[i] = 1
            else:
                yInput[i] = 0

        return cfg1st,pfg1st,literal1st,semantic1st,cfg2nd,pfg2nd,literal2nd,semantic2nd,yInput,pairs

    def GetTestData(self, SELECT=1, PAIRS=[]):
        return self.GetTestEpoch(self.rawData[SELECT*2],self.rawData[SELECT*2+1],PAIRS)

    def GetTestEpoch(self, DATA1ST, DATA2ND, PAIRS):
        pairs = []
        maxNode1st = 0
        maxNode2nd = 0 
        funcNum = len(DATA1ST)
        allPairNum = self.BatchMultiple*len(DATA1ST)
        if PAIRS == []:
            #construct pairs
            for cls1 in range(funcNum):
                #build positive pair
                pairs.append((cls1,cls1))
                maxNode1st = max(maxNode1st, DATA1ST[cls1].nodeNum)
                maxNode2nd = max(maxNode2nd, DATA2ND[cls1].nodeNum)
                #build negetive pair
                for i in range(self.BatchMultiple-1):
                    cls2 = np.random.randint(funcNum)
                    while cls2 == cls1:
                        cls2 = np.random.randint(funcNum)
                    pairs.append((cls1,cls2))        
        else:
            pairs = PAIRS
            for i in range(len(DATA1ST)):
                maxNode1st = max(maxNode1st, DATA1ST[i].nodeNum)
                maxNode2nd = max(maxNode2nd, DATA2ND[i].nodeNum)


        #construct data
        cfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        pfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        pfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        yInput = np.zeros((allPairNum))

        for i in range(allPairNum):
            graph1st = DATA1ST[pairs[i][0]]
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsPFG[u]:
                    pfg1st[i, u, v] = 1
                for l in range(len(graph1st.literal[u])):
                    literal1st[i, u, l] = graph1st.literal[u][l]
                for w in range(len(graph1st.semantic[u])):
                    if w >= self.Semantic1st:
                        break
                    semantic1st[i, u, w] = graph1st.semantic[u][w]
                
            graph2nd = DATA2ND[pairs[i][1]]
            for u in range(graph2nd.nodeNum):
                for v in graph2nd.succsCFG[u]:
                    cfg2nd[i, u, v] = 1
                for v in graph2nd.succsPFG[u]:
                    pfg2nd[i, u, v] = 1
                for l in range(len(graph2nd.literal[u])):
                    literal2nd[i, u, l] = graph2nd.literal[u][l]
                for w in range(len(graph2nd.semantic[u])):
                    if w >= self.Semantic2nd:
                        break
                    semantic2nd[i, u, w] = graph2nd.semantic[u][w]

            if i%self.BatchMultiple == 0:
                yInput[i] = 1
            else:
                yInput[i] = 0

        epochData  = (cfg1st,pfg1st,literal1st,semantic1st,cfg2nd,pfg2nd,literal2nd,semantic2nd,yInput)
        return epochData

    def GetTopkEpoch(self, DATA1ST, DATA2ND):
        pairs = []
        maxNode1st = DATA1ST.nodeNum
        allPairNum = len(DATA2ND)
        maxNode2nd = 0 
        for i in range(allPairNum):
            maxNode2nd = max(maxNode2nd, DATA2ND[i].nodeNum)

        cfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        pfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        pfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        for i in range(allPairNum):
            graph1st = DATA1ST
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsPFG[u]:
                    pfg1st[i, u, v] = 1
                for l in range(len(graph1st.literal[u])):
                    literal1st[i, u, l] = graph1st.literal[u][l]
                for w in range(len(graph1st.semantic[u])):
                    if w >= self.Semantic1st:
                        break
                    semantic1st[i, u, w] = graph1st.semantic[u][w]
                
            graph2nd = DATA2ND[i]
            for u in range(graph2nd.nodeNum):
                for v in graph2nd.succsCFG[u]:
                    cfg2nd[i, u, v] = 1
                for v in graph2nd.succsPFG[u]:
                    pfg2nd[i, u, v] = 1
                for l in range(len(graph2nd.literal[u])):
                    literal2nd[i, u, l] = graph2nd.literal[u][l]
                for w in range(len(graph2nd.semantic[u])):
                    if w >= self.Semantic2nd:
                        break
                    semantic2nd[i, u, w] = graph2nd.semantic[u][w]

        epochData = (cfg1st,pfg1st,literal1st,semantic1st,cfg2nd,pfg2nd,literal2nd,semantic2nd)
        return epochData
