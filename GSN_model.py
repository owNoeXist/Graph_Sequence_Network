import os
import json
import datetime
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import layers,Model

class GSN():
    def __init__(self, HYPER_PARAMETER, LOG_FILE):
        self.model = Model(HYPER_PARAMETER)
        self.HyperParameter = HYPER_PARAMETER
        self.lossObject = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=HYPER_PARAMETER['LearningRate'])
        self.logFile = LOG_FILE

    def name(self):
        return type(self).__name__

    def say(self, STRING):
        print(STRING)
        if self.logFile != None:
            self.logFile.write(string+'\n')

    def save(self, MODEL_PATH, NAME = ""):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        savePath=os.path.join(MODEL_PATH, self.name())
        if NAME != "":
            savePath += NAME
        self.model.save_weights(savePath)
        self.say("Model saved in {}".format(savePath))

    def load(self, MODEL_PATH, NAME = ""):
        savePath=os.path.join(MODEL_PATH, self.name())
        if NAME != "":
            savePath += NAME
        self.model.load_weights(savePath)
        self.say("Model loaded from {}".format(savePath))

    def saveResult(self, RESULT, RESULT_PATH, FILE_NAME="", ):
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        RESULT["HyperParameter"] = self.HyperParameter
        filePath=os.path.join(RESULT_PATH,(self.name()+"_"+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+"_"+FILE_NAME+".json"))
        file=open(filePath,'w')
        file.write("{}\n".format(json.dumps(RESULT)))
        file.close()

    def TrainModel(self, EPOCH_DATA):
        perm = np.random.permutation(len(EPOCH_DATA))
        sumLoss = 0.0

        for index in perm:
            c1, p1, l1 ,s1 ,c2, p2, l2, s2, y = EPOCH_DATA[index]
            with tf.GradientTape() as tape:
                predictions = self.model(c1, p1, l1 ,s1 ,c2, p2, l2, s2)
                batchLoss = self.lossObject(y, predictions)
            gradients = tape.gradient(batchLoss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            sumLoss += batchLoss
        loss = float(sumLoss / len(perm))

        return loss

    def TestModel(self, EPOCH_DATA):
        lossMethod = tf.keras.losses.BinaryCrossentropy()
        testMethod = tf.keras.metrics.AUC()
        c1, p1, l1 ,s1 ,c2, p2, l2, s2, y  = EPOCH_DATA
        predictions = []
        start = 0
        pairNum = len(c1)
        while start < pairNum:
            end = min(start+2000,pairNum)
            predictions += list(self.model(c1[start:end], p1[start:end], l1[start:end], s1[start:end], c2[start:end], p2[start:end], l2[start:end], s2[start:end], TRAINING=False))
            start = end
        loss = float(lossMethod(y, predictions))
        testMethod.update_state(y, predictions)
        auc = float(testMethod.result().numpy())
        return loss,auc

    def GetTopkLabel(self, DATA):
        c1, p1, l1 ,s1 ,c2, p2, l2, s2  = DATA
        label = list(self.model(c1, p1, l1 ,s1 ,c2, p2, l2, s2, TRAINING=False))
        return label

class Model(Model):
    def __init__(self, HYPER_PARAMETER):
        super(Model, self).__init__()
        self.CFGIteraTime = HYPER_PARAMETER['CFGIteraTime']
        self.LFGIteraTime = HYPER_PARAMETER['LFGIteraTime']
        self.EmbedDim = HYPER_PARAMETER['EmbedDim']
        self.DropoutRate = HYPER_PARAMETER['DropoutRate']
        self.OutputDim = HYPER_PARAMETER['OutputDim']
        
        self.data1 = DataLayer(self.EmbedDim, self.DropoutRate)
        self.data2 = DataLayer(self.EmbedDim, self.DropoutRate)
        self.embed = GraphLayer(self.CFGIteraTime, self.LFGIteraTime, self.EmbedDim,self.OutputDim, self.DropoutRate)
        self.outputDense = layers.Dense(self.OutputDim)

    def call(self, CFG1ST, LFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, LFG2ND, LITERAL2ND, SEMANTIC2ND, TRAINING = True):
        literal1st,semantic1st = self.data1.call(LITERAL1ST, SEMANTIC1ST, TRAINING)
        cfg1st = tf.cast(CFG1ST, tf.float32)
        lfg1st = tf.cast(LFG1ST, tf.float32)
        embed1st = self.embed.call(literal1st, semantic1st, cfg1st, lfg1st, TRAINING)

        literal1st,semantic2nd = self.data2.call(LITERAL2ND, SEMANTIC2ND, TRAINING)
        cfg2nd = tf.cast(CFG2ND, tf.float32)
        lfg2nd = tf.cast(LFG2ND, tf.float32)
        embed2nd = self.embed.call(literal1st, semantic2nd, cfg2nd, lfg2nd, TRAINING)

        cosin = tf.keras.losses.cosine_similarity(embed1st, embed2nd, axis=1)
        label = tf.divide(-tf.subtract(cosin,1),2)

        return label

class DataLayer(layers.Layer):
    def __init__(self, EMBED_DIM, DROPOUT_RATE):
        super(DataLayer, self).__init__()
        self.EmbedDim = EMBED_DIM
        self.DropoutRate = DROPOUT_RATE

        self.literalDense = layers.Dense(self.EmbedDim,activation=None,use_bias=False)
        self.semanticGRU = layers.GRU(self.EmbedDim, activation='tanh', dropout=self.DropoutRate)
    
    def call(self, LITERAL, SEMANTIC, TRAINING = True):
        #literal: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        literal = self.literalDense(LITERAL)
        #sematic: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        semantic= self.semanticGRU(SEMANTIC, training=TRAINING)
        return literal,semantic

class GraphLayer(layers.Layer):
    def __init__(self, CFG_ITERA_TIME, LFG_ITERA_TIME, EMBED_DIM, OUTPUT_DIM, DROPOUT_RATE):
        super(GraphLayer, self).__init__()
        self.CFGIteraTime = CFG_ITERA_TIME
        self.LFGIteraTime = LFG_ITERA_TIME
        self.EmbedDim = EMBED_DIM
        self.OutputDim = OUTPUT_DIM
        self.DropoutRate = DROPOUT_RATE
        
        self.cfgAttention = AttentionLayer()
        self.lfgAttention = AttentionLayer()
        self.MLPs = []
        self.MLPs.append(layers.Dense(self.EmbedDim, activation='relu', use_bias=False))
        self.MLPs.append(layers.Dense(self.EmbedDim, use_bias=False))
        self.outputDense = layers.Dense(self.OutputDim)
        #self.Dropout = layers.Dropout(self.DropoutRate)
    
    def call(self, LITERAL, SEMANTIC, CFG, LFG, TRAINING = True):
        #Embedding by cfg
        #cfgEmbed: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        cfgEmbed = tf.nn.relu(LITERAL)
        for _ in range(self.CFGIteraTime):
            #Update attention
            #attention: [BATCH_SIZE, NODE_NUM, NODE_NUM]
            attention = self.cfgAttention(cfgEmbed, CFG)
            #Message pass
            cfgEmbed = tf.matmul(attention, cfgEmbed)
            for mlp in self.MLPs:
                cfgEmbed = mlp(cfgEmbed)
            #transMid = self.Dropout(transMid, training=TRAINING)
            #Adding and Nonlinearity
            cfgEmbed = tf.nn.tanh(LITERAL + cfgEmbed)
        #Embedding by lfg
        #lfgEmbed: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        lfgEmbed = tf.nn.relu(LITERAL)
        for _ in range(self.LFGIteraTime):
            #Update attention
            #attention: [BATCH_SIZE, NODE_NUM, NODE_NUM]
            attention = self.lfgAttention(lfgEmbed, LFG)
            #Message pass
            lfgEmbed = tf.matmul(attention, lfgEmbed)
            for mlp in self.MLPs:
                lfgEmbed = mlp(lfgEmbed)
            #transMid = self.Dropout(transMid, training=TRAINING)
            #Adding and Nonlinearity
            lfgEmbed = tf.nn.tanh(LITERAL + lfgEmbed)
        #Combine Embedding
        #output: [BATCH_SIZE, EMBED_DIM]
        midOutput = tf.reduce_sum(tf.concat([cfgEmbed,lfgEmbed],-1), 1)
        midOutput = tf.concat([midOutput,SEMANTIC],-1)
        output = self.outputDense(midOutput)
        return output

class AttentionLayer(layers.Layer):
  def __init__(self):    
    super(AttentionLayer,self).__init__()
    self.aWeight1 = layers.Dense(1)
    self.aWeight2 = layers.Dense(1) 
    
  def __call__(self, FEATURE, GRAPH):
    #wh: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
    wh = FEATURE
    #ah1: [BATCH_SIZE, NODE_NUM, 1]
    ah1 = self.aWeight1(wh)
    #ah2: [BATCH_SIZE, NODE_NUM, 1]
    ah2 = self.aWeight2(wh)
    #eij: [BATCH_SIZE, NODE_NUM, NODE_NUM]
    eij = ah1 + tf.transpose(ah2,[0,2,1])
    #aij: [BATCH_SIZE, NODE_NUM, NODE_NUM]
    eGraph = tf.subtract(tf.multiply(GRAPH,1e20),1e20)
    aij = tf.nn.softmax(tf.nn.leaky_relu(eij)+eGraph)
    #Reduce Calculate
    aij = tf.multiply(aij,GRAPH)

    return aij