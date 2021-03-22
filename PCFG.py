import numpy as np

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

    def IsSame(self, PCFG_CMP):
        if self.name == PCFG_CMP.name and \
             self.nodeNum == PCFG_CMP.nodeNum and \
                 self.succsCFG == PCFG_CMP.succsCFG and \
                     len(self.literal[0])  == len(PCFG_CMP.literal[1]) and \
                         len(self.semantic[0]) == len(PCFG_CMP.semantic[0]):
                         return True
        return False
