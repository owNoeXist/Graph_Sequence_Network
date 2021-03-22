DataPath    = './Data'
ModelPath   = './Model'
ResultPath  = './Result'
LogPath     = None
Partitions  = [0.8,0.1,0.1]
TrainEpoch  = 100
TestFreq    = 5
SaveFreq    = 50
Hyper       = {
    'Literal1st':   6,
    'Semantic1st':  200,
    'Literal2nd':   5,
    'Semantic2nd':  200,
    'CFGForwardTime':   4,
    'CFGReverseTime':   2,
    'PFGIteraTime': 2,
    'MLPLayer':   2,
    'EmbedDim':     64,
    'DropoutRate':  0.4,
    'OutputDim':    64,
    'BatchSize':    8,
    'BatchMultiple':2,
    'LearningRate': 0.0001
}
