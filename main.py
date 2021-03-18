'''
======================================================================
Cross-Class-Language Similarity Compare
|----------------------------------|
|Author         |Yuan Wei          |
|Setup Time     |2019.11.11        |
|Version        |V1.0              |
|Update Time    |2020.11.16        |
|----------------------------------|
======================================================================
This model has three part：
1. Data Processing
2. Public Model Method
3. Specific Model
======================================================================
Update Log：
2020.11.16  ——  Reconstruct Code Struct，divide function into file
======================================================================
'''
import os
import tensorflow as tf
from  datetime import datetime

import train
import CEC_parameters as param
from CEC_data import MyData
from CEC_model import CEC

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    datas = MyData(param.Hyper,param.DataPath,param.Partitions)
    trainData = datas.GetTrainData()
    validData = datas.GetValidData()

    model = CEC(param.Hyper)

    train.TrainModel(model,trainData,validData,param.ModelPath,param.TrainEpoch,param.TestFreq,param.SaveFreq)
    
    
    #LOAD_PATH = os.path.join(param.ModelPath, mainModel.name())
    #LOAD_PATH = None
    #mainModel.init(LOAD_PATH, param.LogPath)
    #mainModel.GetTopK(dataTrainX, dataTrainY, 10)