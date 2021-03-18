class CFG(object):
    def __init__(self, funcname=None, nodenum=0):
        self.name = funcname
        self.nodeNum = nodenum
        self.succs = []
        self.preds = []
        self.basic = []
        for _ in range(nodenum):
            self.succs.append([])
            self.preds.append([])
            self.basic.append([])
        
    def AddDirectedEdge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def AddUndirectedEdge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)
        self.succs[v].append(u)
        self.preds[u].append(v)

'''
def NodeFold(GRAPH):
    i=1
    while i < GRAPH['Nodenum']:
        #CheckNode
        if GRAPH['NodeFeature'][i][0] <= 2 and i in GRAPH['NodeTo'][i-1]:
            if len(GRAPH['NodeTo'][i])==2 and len(GRAPH['NodeTo'][i-1])==2:
                for j in range(GRAPH['Nodenum']):
                    if i in GRAPH['NodeTo'][j] and j!=i-1:
                        break
                if j != GRAPH['Nodenum']:
                    i+=1
                    continue
                sameNode = [x for x in GRAPH['NodeTo'][i-1] if x in GRAPH['NodeTo'][i]]
                if sameNode is None:
                    i+=1
                    continue
            else:
                i+=1
                continue
        else:        
            i+=1
            continue
        #DeleteNode
        for j in range(len(GRAPH['NodeFeature'][0])):
            GRAPH['NodeFeature'][i-1][j]+=GRAPH['NodeFeature'][i][j]
        del GRAPH['NodeFeature'][i]
        for j in range(2):
            GRAPH['NodeTo'][i-1][j]=GRAPH['NodeTo'][i][j]
        del GRAPH['NodeTo'][i]
        GRAPH['Nodenum']-=1
        for j in range(GRAPH['Nodenum']):
            for k in range(len(GRAPH['NodeTo'][j])):
                if GRAPH['NodeTo'][j][k] > i:
                    GRAPH['NodeTo'][j][k]-=1
    return GRAPH
'''