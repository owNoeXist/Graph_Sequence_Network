DataPath    = './Data'
ModelPath   = './Model'
ResultPath  = './Result'
LogPath     = None
Partitions  = [0.8,0.1,0.05]
TrainEpoch  = 50
TestFreq    = 5
SaveFreq    = 50
Hyper       = {
    'File1st':      'DataC.json',
    'Literal1st':   7,
    'Semantic1st':  50,
    'File2nd':      'DataX86-O0.json',
    'Literal2nd':   7,
    'Semantic2nd':  50,
    'CFGIteraTime':   5,
    'LFGIteraTime': 1,
    'EmbedDim':     64,
    'OutputDim':    64,
    'DropoutRate':  0.9,
    'BatchSize':    32,
    'BatchMultiple':2,
    'LearningRate': 0.001
}
