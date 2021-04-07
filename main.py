'''
======================================================================
Graph Sequence Network
|----------------------------------|
|Author         |Yuan Wei          |
|----------------------------------|
======================================================================
'''
import os
import tensorflow as tf
from  datetime import datetime

import train
import GSN_parameters as param
from GSN_data import GSNData
from GSN_model import GSN

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(gpus)
    
    datas = GSNData(param.Hyper,param.DataPath,param.Partitions)
    model = GSN(param.Hyper, param.LogPath)
    #model.load(param.ModelPath)

    trainData, pairs = datas.GetTrainData(SELECT=0)
    trainTest = datas.GetTestData(SELECT=0, PAIRS=pairs)
    validTest = datas.GetTestData(SELECT=1)
    
    train.TrainModel(model,trainData,trainTest,validTest,param.ModelPath,param.TrainEpoch,param.TestFreq,param.SaveFreq,param.ResultPath)
    
    #model.load(param.ModelPath)
    model.load(param.ModelPath,"_best")
    train.GetTopSim(model, datas, param.ResultPath, SELECT=2, K=10)

    #train.MatchFunction(model, datas, param.ResultPath, SELECT=2)
