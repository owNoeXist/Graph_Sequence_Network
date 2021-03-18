class PCFG(object):
    def __init__(self, funcname=None, nodenum=0):
        self.name     = funcname
        self.nodeNum  = nodenum
        self.succsCFG = [[]]*nodenum
        self.predsCFG = [[]]*nodenum
        self.succsPFG = [[]]*nodenum
        self.predsPFG = [[]]*nodenum
        self.literal  = [[]]*nodenum
        self.semantic = [[]]*nodenum
        
    def AddDirectedEdgeCFG(self, u, v):
        self.succsCFG[u].append(v)
        self.predsCFG[v].append(u)

    def AddDirectedEdgePFG(self, u, v):
        self.succsPFG[u].append(v)
        self.predsPFG[v].append(u)

    def AddUndirectedEdge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)
        self.succs[v].append(u)
        self.preds[u].append(v)