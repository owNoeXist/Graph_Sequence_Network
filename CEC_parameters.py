DataPath    = './Data'
ModelPath   = './Model'
ResultPath  = './Result'
LogPath     = None
Partitions  = [0.8,0.1,0.1]
TrainEpoch  = 100
TestFreq    = 5
SaveFreq    = 50
Hyper       = {
    'Word1st':      20,
    'Line1st':      30,
    'Word2nd':      4,
    'Line2nd':      300,
    'IteraTimes':   5,
    'EmbedLayer':   2,
    'EmbedDim':     64,
    'OutputDim':    64,
    'BatchSize':    5,
    'LearningRate': 1e-4
}
