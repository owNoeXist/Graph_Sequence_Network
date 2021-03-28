DataPath    = './Data'
ModelPath   = './Model'
ResultPath  = './Result'
LogPath     = None
Partitions  = [0.8,0.1,0.05]
TrainEpoch  = 50
TestFreq    = 5
SaveFreq    = 20
Hyper       = {
    'Literal1st':   7,
    'Semantic1st':  200,
    'Literal2nd':   7,
    'Semantic2nd':  200,
    'CFGIteraTime':   5,
    'LFGIteraTime': 2,
    'EmbedDim':     64,
    'OutputDim':    64,
    'DropoutRate':  0.9,
    'BatchSize':    32,
    'BatchMultiple':2,
    'LearningRate': 0.001
}
