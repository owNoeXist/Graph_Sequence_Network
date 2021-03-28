import os
import json
import numpy as np

from CFG import CFG

class GSNData():
    def __init__(self, HYPER_PARAMETER, DATA_PATH, PARTITIONS):
        self.Literal1st = HYPER_PARAMETER['Literal1st']
        self.Semantic1st = HYPER_PARAMETER['Semantic1st']
        self.Literal2nd = HYPER_PARAMETER['Literal2nd']
        self.Semantic2nd = HYPER_PARAMETER['Semantic2nd']
        self.BatchSize = HYPER_PARAMETER['BatchSize']
        self.BatchMultiple = HYPER_PARAMETER['BatchMultiple']
        self.rawData = self.ObtainRawData(DATA_PATH,PARTITIONS)

    #================================================================
    def ObtainRawData(self, DATA_PATH, PARTITIONS):
        #Go through software folder and exact function into class
        data1st = []
        data2nd = []
        for dir in os.listdir(DATA_PATH):
            programPath = os.path.join(DATA_PATH,dir)
            if not os.path.isdir(programPath):
                continue
            for file in os.listdir(programPath):
                data = []
                filePath = os.path.join(programPath,file)
                with open(filePath) as lines:
                    for line in lines:
                        data.append(self.exactData(dir, line))
                data = self.clearData(data)
                if file not in ['DataC.json']:
                    data1st.append(data)
                else:
                    data2nd.append(data)
            print("{0}:\t{1} X86 functions, {2} C functions".format(dir,len(data1st[-1]),len(data2nd[-1])))
        data1st, data2nd = self.matchFunction(data1st, data2nd)
        dataPart = self.partData(data1st, data2nd, DATA_PATH, PARTITIONS)
        return dataPart

    def exactData(self, DIR, LINE):
        graphInfo = json.loads(LINE.strip())
        curCFG = CFG(DIR+'-'+graphInfo['Funcname'], graphInfo['Nodenum'])
        for u in range(graphInfo['Nodenum']):
            for v in graphInfo['CFG'][u]:
                curCFG.AddDirectedEdgeCFG(u, v)
            curCFG.literal[u]  = np.array(graphInfo['Literal'][u], dtype=object)
            curCFG.semantic[u] = np.array(graphInfo['Semantic'][u], dtype=object)
        curCFG = self.levelFlowGraph(curCFG)
        return curCFG

    def levelFlowGraph(self, CFG):
        nodelevel = [-1]*CFG.nodeNum
        level = 1
        queue = [0]
        nodelevel[0] = 0
        while len(queue):
            if nodelevel[queue[0]]>=level:
                level+=1
            for i in range(len(CFG.succsCFG[queue[0]])):
                if nodelevel[CFG.succsCFG[queue[0]][i]] == -1:
                    queue.append(CFG.succsCFG[queue[0]][i])
                    nodelevel[CFG.succsCFG[queue[0]][i]] = level
            del queue[0]
        preLevel = [0]
        nowLevel = []
        for i in range(1,level):
            for j in range(1,CFG.nodeNum):
                if nodelevel[j] == i:
                    for node in preLevel:
                        CFG.AddDirectedEdgeLFG(j,node)
                    nowLevel.append(j)
            preLevel = nowLevel
            nowLevel = []
        return CFG

    def clearData(self, DATA):
        DATA.sort(key=lambda CFG:CFG.name)
        i=0
        while i<len(DATA)-1:
            if DATA[i].name == DATA[i+1].name:
                if DATA[i].IsSame(DATA[i+1]):
                    DATA[i:]=DATA[i+1:]
                else:    
                    j=2
                    while i+j < len(DATA) and DATA[i].name == DATA[i+j].name:
                        j+=1
                    DATA[i:]=DATA[i+j:]
            else:
                i+=1
        return DATA

    def matchFunction(self, DATA1ST, DATA2ND):
        for i in range(len(DATA1ST)):
            j=0
            while j < len(DATA1ST[i]) and j < len(DATA2ND[i]):
                if DATA1ST[i][j].name==DATA2ND[i][j].name:
                    if DATA1ST[i][j].nodeNum <= 100 and DATA2ND[i][j].nodeNum <= 100 \
                        and 0.5 <= DATA1ST[i][j].nodeNum/DATA2ND[i][j].nodeNum <= 2:
                        j+=1
                    else:
                        DATA1ST[i][j:]=DATA1ST[i][j+1:]
                        DATA2ND[i][j:]=DATA2ND[i][j+1:]
                elif DATA1ST[i][j].name < DATA2ND[i][j].name:
                    DATA1ST[i][j:]=DATA1ST[i][j+1:]
                else:
                    DATA2ND[i][j:]=DATA2ND[i][j+1:]
            if j < len(DATA1ST[i]):
                DATA1ST[i][j:]=[]
            if j < len(DATA2ND[i]):
                DATA2ND[i][j:]=[]
            print("{0} matched functions".format(len(DATA1ST[i])))
        return DATA1ST, DATA2ND

    def partData(self, DATA1ST, DATA2ND, DATA_PATH, PARTITIONS):
        #Load perm file to use before data part
        dataPart1st = []
        dataPart2nd = []
        for i in range(len(DATA1ST)):
            dataPart1st.extend(DATA1ST[i])
            dataPart2nd.extend(DATA2ND[i])
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

    #================================================================
    def GetTrainData(self,SELECT=0):
        return self.getTrainEpoch(self.rawData[SELECT*2],self.rawData[SELECT*2+1])

    def getTrainEpoch(self, DATA1ST, DATA2ND):
        start = 0
        pairs = []
        epochData = []
        funcNum = len(DATA1ST)
        while start < funcNum:
            c1, e1, l1 ,s1 ,c2, e2, l2, s2, y, pair = self.getTrainPair(DATA1ST, DATA2ND, START=start)
            epochData.append((c1, e1, l1 ,s1 ,c2, e2, l2, s2, y))
            pairs += pair
            start += self.BatchSize
        return epochData,pairs

    def getTrainPair(self, DATA1ST, DATA2ND, START = -1):
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
                while cls2 == cls1 or 0.8 < DATA1ST[cls1].nodeNum/DATA2ND[cls2].nodeNum < 1.25:
                    cls2 = np.random.randint(funcNum)
                pairs.append((cls1,cls2))
                maxNode2nd = max(maxNode2nd, DATA2ND[cls2].nodeNum)

        cfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        lfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        lfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        yInput = np.zeros((allPairNum))

        for i in range(allPairNum):
            graph1st = DATA1ST[pairs[i][0]]
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsLFG[u]:
                    lfg1st[i, u, v] = 1
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
                for v in graph2nd.succsLFG[u]:
                    lfg2nd[i, u, v] = 1
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

        return cfg1st,lfg1st,literal1st,semantic1st,cfg2nd,lfg2nd,literal2nd,semantic2nd,yInput,pairs

    #================================================================
    def GetTestData(self, SELECT=1, PAIRS=[]):
        return self.getTestEpoch(self.rawData[SELECT*2],self.rawData[SELECT*2+1],PAIRS)

    def getTestEpoch(self, DATA1ST, DATA2ND, PAIRS):
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
        lfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        lfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        yInput = np.zeros((allPairNum))

        for i in range(allPairNum):
            graph1st = DATA1ST[pairs[i][0]]
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsLFG[u]:
                    lfg1st[i, u, v] = 1
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
                for v in graph2nd.succsLFG[u]:
                    lfg2nd[i, u, v] = 1
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

        epochData  = (cfg1st,lfg1st,literal1st,semantic1st,cfg2nd,lfg2nd,literal2nd,semantic2nd,yInput)
        return epochData
    
    #================================================================
    def GetTopkEpoch(self, DATA1ST, DATA2ND):
        pairs = []
        maxNode1st = DATA1ST.nodeNum
        allPairNum = len(DATA2ND)
        maxNode2nd = 0 
        for i in range(allPairNum):
            maxNode2nd = max(maxNode2nd, DATA2ND[i].nodeNum)

        cfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        lfg1st = np.zeros((allPairNum, maxNode1st, maxNode1st))
        literal1st = np.zeros((allPairNum, maxNode1st, self.Literal1st))
        semantic1st = np.zeros((allPairNum, maxNode1st, self.Semantic1st))

        cfg2nd = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        lfg2nd  = np.zeros((allPairNum, maxNode2nd, maxNode2nd))
        literal2nd  = np.zeros((allPairNum, maxNode2nd, self.Literal2nd))
        semantic2nd  = np.zeros((allPairNum, maxNode2nd, self.Semantic2nd))

        for i in range(allPairNum):
            graph1st = DATA1ST
            for u in range(graph1st.nodeNum):
                for v in graph1st.succsCFG[u]:
                    cfg1st[i, u, v] = 1
                for v in graph1st.succsLFG[u]:
                    lfg1st[i, u, v] = 1
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
                for v in graph2nd.succsLFG[u]:
                    lfg2nd[i, u, v] = 1
                for l in range(len(graph2nd.literal[u])):
                    literal2nd[i, u, l] = graph2nd.literal[u][l]
                for w in range(len(graph2nd.semantic[u])):
                    if w >= self.Semantic2nd:
                        break
                    semantic2nd[i, u, w] = graph2nd.semantic[u][w]

        epochData = (cfg1st,lfg1st,literal1st,semantic1st,cfg2nd,lfg2nd,literal2nd,semantic2nd)
        return epochData
