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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(gpus)
    
    datas = MyData(param.Hyper,param.DataPath,param.Partitions)
    model = GSN(param.Hyper)

    trainData, pairs = datas.GetTrainData(SELECT=0)
    trainTest = datas.GetTestData(SELECT=0, PAIRS=pairs)
    validTest = datas.GetTestData(SELECT=1)
    train.TrainModel(model,trainData,trainTest,validTest,param.ModelPath,param.TrainEpoch,param.TestFreq,param.SaveFreq,param.ResultPath)
    
    #model.load(param.ModelPath)

    train.GetTopK(model, datas, param.ResultPath, SELECT=1, K=10)
    #train.MatchFunction(model, datas, param.ResultPath)
