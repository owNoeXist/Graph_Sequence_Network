class PCFG(object):
    def __init__(self, funcname=None, nodenum=0):
        self.name     = funcname
        self.nodeNum  = nodenum
        self.succsCFG = []
        self.predsCFG = []
        self.succsPFG = []
        self.predsPFG = []
        self.literal  = []
        self.semantic = []
        for i in range(nodenum):
            self.succsCFG.append([])
            self.predsCFG.append([])
            self.succsPFG.append([])
            self.predsPFG.append([])
            self.literal.append([])
            self.semantic.append([])
        
    def AddDirectedEdgeCFG(self, u, v):
        self.succsCFG[u].append(v)
        self.predsCFG[v].append(u)

    def AddDirectedEdgePFG(self, u, v):
        self.succsPFG[u].append(v)
        self.predsPFG[v].append(u)