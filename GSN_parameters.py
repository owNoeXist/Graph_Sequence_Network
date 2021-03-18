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
    'Semantic1st':   200,
    'Literal2nd':   5,
    'Semantic2nd':   200,
    'IteraTimes':   5,
    'EmbedLayer':   2,
    'EmbedDim':     64,
    'OutputDim':    64,
    'BatchSize':    5,
    'Multiple':     2,
    'LearningRate': 1e-4
}
