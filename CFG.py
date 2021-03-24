import numpy as np

class CFG(object):
    def __init__(self, funcname=None, nodenum=0):
        self.name     = funcname
        self.nodeNum  = nodenum
        self.succsCFG = []
        self.predsCFG = []
        self.succsLFG = []
        self.predsLFG = []
        self.literal  = []
        self.semantic = []
        for i in range(nodenum):
            self.succsCFG.append([])
            self.predsCFG.append([])
            self.succsLFG.append([])
            self.predsLFG.append([])
            self.literal.append([])
            self.semantic.append([])

    def AddDirectedEdgeCFG(self, u, v):
        self.succsCFG[u].append(v)
        self.predsCFG[v].append(u)

    def AddDirectedEdgeLFG(self, u, v):
        self.succsLFG[u].append(v)
        self.predsLFG[v].append(u)

    def IsSame(self, CFG_CMP):
        if self.name == CFG_CMP.name and \
             self.nodeNum == CFG_CMP.nodeNum and \
                 self.succsCFG == CFG_CMP.succsCFG and \
                     len(self.literal[0])  == len(CFG_CMP.literal[1]) and \
                         len(self.semantic[0]) == len(CFG_CMP.semantic[0]):
                         return True
        return False
