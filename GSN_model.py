import os
import json
import datetime
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import layers,Model

class GSN():
    def __init__(self, HYPER_PARAMETER, LOG_FILE):
        self.model = MyModel(HYPER_PARAMETER)
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

class MyModel(Model):
    def __init__(self, HYPER_PARAMETER):
        self.CFGForwardTime = HYPER_PARAMETER['CFGForwardTime']
        self.CFGReverseTime = HYPER_PARAMETER['CFGReverseTime']
        self.LFGIteraTime = HYPER_PARAMETER['LFGIteraTime']
        self.EmbedDim = HYPER_PARAMETER['EmbedDim']
        self.DropoutRate = HYPER_PARAMETER['DropoutRate']
        self.OutputDim = HYPER_PARAMETER['OutputDim']
        
        super(MyModel, self).__init__()
        self.data1 = MyDataLayer(self.EmbedDim, self.DropoutRate)
        self.data2 = MyDataLayer(self.EmbedDim, self.DropoutRate)
        self.embed = MyGraphLayer(self.CFGForwardTime, self.CFGReverseTime, self.LFGIteraTime, self.EmbedDim,self.OutputDim, self.DropoutRate)
        
    def call(self, CFG1ST, LFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, LFG2ND, LITERAL2ND, SEMANTIC2ND, TRAINING = True):
        basicBlock1st = self.data1.call(LITERAL1ST, SEMANTIC1ST, TRAINING)
        cfg1st = tf.cast(CFG1ST, tf.float32)
        lfg1st = tf.cast(LFG1ST, tf.float32)
        embed1st = self.embed.call(basicBlock1st, cfg1st, lfg1st, TRAINING)

        basicBlock2nd = self.data2.call(LITERAL2ND, SEMANTIC2ND, TRAINING)
        cfg2nd = tf.cast(CFG2ND, tf.float32)
        lfg2nd = tf.cast(LFG2ND, tf.float32)
        embed2nd = self.embed.call(basicBlock2nd, cfg2nd, lfg2nd, TRAINING)

        cosin = tf.keras.losses.cosine_similarity(embed1st, embed2nd, axis=1)
        label = tf.divide(-tf.subtract(cosin,1),2)

        return label

class MyDataLayer(layers.Layer):
    def __init__(self, EMBED_DIM, DROPOUT_RATE):
        super(MyDataLayer, self).__init__()
        self.EmbedDim = EMBED_DIM
        self.HalfDim = int(EMBED_DIM/2)
        self.DropoutRate = DROPOUT_RATE

        self.Dense = layers.Dense(self.HalfDim,activation='relu',use_bias=False)
        self.DenseDropout = layers.Dropout(self.DropoutRate)
        self.GRU = layers.GRU(self.HalfDim, activation='tanh', return_sequences=True, return_state=True, dropout=self.DropoutRate)
    
    def call(self, LITERAL, SEMANTIC, TRAINING = True):
        literal = self.DenseDropout(self.Dense(LITERAL), training=TRAINING)
        sematic,fs = self.GRU(SEMANTIC, training=TRAINING)
        #output: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        output = tf.concat([literal,sematic],-1)
        return output

class MyGraphLayer(layers.Layer):
    def __init__(self, CFG_FORWARD_TIME, CFG_REVERSE_TIME, LFG_ITERA_TIME, EMBED_DIM, OUTPUT_DIM, DROPOUT_RATE):
        super(MyGraphLayer, self).__init__()
        self.CFGForwardTime = CFG_FORWARD_TIME
        self.CFGReverseTime = CFG_REVERSE_TIME
        self.LFGIteraTime = LFG_ITERA_TIME
        self.DropoutRate = DROPOUT_RATE
        
        self.CFGAttention = MyAttentionLayer(EMBED_DIM, DROPOUT_RATE)
        self.weightDense = layers.Dense(EMBED_DIM, use_bias=False)
        self.Dropout = layers.Dropout(self.DropoutRate)
        self.outputDense = layers.Dense(OUTPUT_DIM)
    
    def call(self, INPUT, CFG, LFG, TRAINING = True):
        #Embedding by cfg
        #cfgForwardEmbed: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        cfgForwardEmbed = tf.nn.relu(INPUT)
        for _ in range(self.CFGForwardTime):
            #Update attention
            #attention: [BATCH_SIZE, NODE_NUM, NODE_NUM]
            attention = self.CFGAttention(cfgForwardEmbed, CFG)
            attention = tf.multiply(attention,CFG)
            #Message pass
            #transMid: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
            transMid = self.weightDense(cfgForwardEmbed)
            transMid = tf.matmul(attention, transMid)
            transMid = self.Dropout(transMid, training=TRAINING)
            #Adding and Nonlinearity
            cfgForwardEmbed = tf.nn.elu(cfgForwardEmbed + transMid)
        
        #Embedding by reversed cfg
        #cfgReverseEmbed: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
        cfgReverseEmbed = tf.nn.relu(INPUT)
        CFGReverse = np.transpose(CFG,(0,2,1))
        for _ in range(self.CFGReverseTime):
            #Update attention
            #attention: [BATCH_SIZE, NODE_NUM, NODE_NUM]
            attention = self.CFGAttention(cfgReverseEmbed, CFGReverse)
            attention = tf.multiply(attention,CFGReverse)
            #Message pass
            #transMid: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
            transMid = self.weightDense(cfgReverseEmbed)
            transMid = tf.matmul(attention, transMid)
            transMid = self.Dropout(transMid, training=TRAINING)
            #Adding and Nonlinearity
            cfgReverseEmbed = tf.nn.elu(cfgReverseEmbed + transMid)
        '''
        #Embedding by lfg
        lfgEmbed = tf.nn.relu(INPUT)
        for _ in range(self.LFGIteraTime):
            #Message convey
            transMid = tf.matmul(LFG, lfgEmbed)
            #Weight calculation
            for dense in self.mlpLFG:
                transMid = dense(transMid)
            #Adding and Nonlinearity
            lfgEmbed = tf.nn.tanh(lfgEmbed + transMid)
        
        #dropOutput = self.Dropout(cfgForwardEmbed+cfgReverseEmbed)
        '''
        #midOutput: [BATCH_SIZE, EMBED_DIM]
        midOutput = tf.reduce_sum(cfgForwardEmbed+cfgReverseEmbed, 1)
        output = self.outputDense(midOutput)
    
        return output

class MyAttentionLayer(layers.Layer):
  def __init__(self, EMBED_DIM, DROPOUT_RATE):    
    super(MyAttentionLayer,self).__init__()    
    self.EmbedDim = EMBED_DIM
    self.aWeightDense1 = layers.Dense(1)
    self.aWeightDense2 = layers.Dense(1) 
    
  def __call__(self, INPUT, GRAPH):
    #wh: [BATCH_SIZE, NODE_NUM, EMBED_DIM]
    wh = INPUT
    #ah1: [BATCH_SIZE, NODE_NUM, 1]
    ah1 = self.aWeightDense1(wh)
    #ah2: [BATCH_SIZE, NODE_NUM, 1]
    ah2 = self.aWeightDense2(wh)
    #eij: [BATCH_SIZE, NODE_NUM, NODE_NUM]
    eij = ah1 + tf.transpose(ah2,[0,2,1])
    # aij: [BATCH_SIZE, NODE_NUM, NODE_NUM]
    eGraph = tf.subtract(tf.multiply(GRAPH,1e20),1e20)
    aij = tf.nn.softmax(tf.nn.leaky_relu(eij)+eGraph)

    return aij