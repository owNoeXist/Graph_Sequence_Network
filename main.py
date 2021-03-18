'''
======================================================================
Graph Sequence Network
|----------------------------------|
|Author         |Yuan Wei          |
|Version        |V1.0              |
|Update Time    |2021.03.18        |
|----------------------------------|
======================================================================
This model has three part：
1. Data Processing
2. Public Model Method
3. Specific Model
======================================================================
Update Log：
2021.xx.xx  ——  
======================================================================
'''
import os
import tensorflow as tf
from  datetime import datetime

import train
import GSN_parameters as param
from GSN_data import MyData
from GSN_model import GSN
from PCFG import PCFG

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(gpus)
    
    datas = MyData(param.Hyper,param.DataPath,param.Partitions)
    model = GSN(param.Hyper)

    trainData = datas.GetTrainData()
    validData = datas.GetValidData()
    train.TrainModel(model,trainData,validData,param.ModelPath,param.TrainEpoch,param.TestFreq,param.SaveFreq)
    
    
    #LOAD_PATH = os.path.join(param.ModelPath, mainModel.name())
    #LOAD_PATH = None
    #mainModel.init(LOAD_PATH, param.LogPath)
    #mainModel.GetTopK(dataTrainX, dataTrainY, 10)