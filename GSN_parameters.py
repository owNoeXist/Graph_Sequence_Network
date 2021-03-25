DataPath    = './Data'
ModelPath   = './Model'
ResultPath  = './Result'
LogPath     = None
Partitions  = [0.8,0.1,0.05]
TrainEpoch  = 0
TestFreq    = 5
SaveFreq    = 20
Hyper       = {
    'Literal1st':   7,
    'Semantic1st':  200,
    'Literal2nd':   7,
    'Semantic2nd':  300,
    'CFGForwardTime':   3,
    'CFGReverseTime':   1,
    'LFGIteraTime': 1,
    'MLPLayer':   2,
    'EmbedDim':     64,
    'DropoutRate':  0.6,
    'OutputDim':    64,
    'BatchSize':    8,
    'BatchMultiple':2,
    'LearningRate': 0.0001
}
