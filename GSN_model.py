import os
import json
import datetime
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow.keras import layers,Model
from sklearn.metrics import auc, roc_curve, precision_recall_curve

class GSN():
    def __init__(self, HYPER_PARAMETER):
        self.model = MyModel(HYPER_PARAMETER)
        self.HyperParameter = HYPER_PARAMETER
        self.lossObject = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=HYPER_PARAMETER['LearningRate'])
        self.logFile = None

    def name(self):
        return type(self).__name__

    def say(self, STRING):
        print(STRING)
        if self.logFile != None:
            self.logFile.write(string+'\n')

    def save(self, MODEL_PATH, NAME = None):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        savePath=os.path.join(MODEL_PATH, self.name())
        if NAME != None:
            savePath += NAME
        self.model.save_weights(savePath)
        self.say("Model saved in {}".format(savePath))

    def load(self, MODEL_PATH, NAME = False):
        savePath=os.path.join(MODEL_PATH, self.name())
        if NAME == True:
            savePath += NAME
        self.model.load_weights(savePath)

    def saveResult(self, RESULT, RESULT_PATH, FILE_NAME="", ):
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
        loss = sumLoss / len(perm)

        return loss

    def TestModel(self, EPOCH_DATA):
        testMethod = tf.keras.metrics.AUC()
        c1, p1, l1 ,s1 ,c2, p2, l2, s2, y  = EPOCH_DATA
        predictions = self.model(c1, p1, l1 ,s1 ,c2, p2, l2, s2)
        testMethod.update_state(y, predictions)
        testResult = float(testMethod.result().numpy())
        return testResult

    def GetTopkLabel(self, DATA):
        c1, p1, l1 ,s1 ,c2, p2, l2, s2  = DATA
        label = list(self.model(c1, p1, l1 ,s1 ,c2, p2, l2, s2))
        return label

class MyModel(Model):
    def __init__(self, HYPER_PARAMETER):
        self.Literal1st = HYPER_PARAMETER['Literal1st']
        self.Semantic1st = HYPER_PARAMETER['Semantic1st']
        self.Literal2nd = HYPER_PARAMETER['Literal2nd']
        self.Semantic2nd = HYPER_PARAMETER['Semantic2nd']
        self.CFGForwardTime = HYPER_PARAMETER['CFGForwardTime']
        self.CFGReverseTime = HYPER_PARAMETER['CFGReverseTime']
        self.PFGIteraTime = HYPER_PARAMETER['PFGIteraTime']
        self.MLPLayer = HYPER_PARAMETER['MLPLayer']
        self.EmbedDim = HYPER_PARAMETER['EmbedDim']
        self.DropoutRate = HYPER_PARAMETER['DropoutRate']
        self.OutputDim = HYPER_PARAMETER['OutputDim']
        
        super(MyModel, self).__init__()
        self.data1 = MyDataLayer( self.Literal1st, self.Semantic1st, self.EmbedDim, self.DropoutRate)
        self.data2 = MyDataLayer( self.Literal2nd, self.Semantic2nd, self.EmbedDim, self.DropoutRate)
        self.embed = MyGraphLayer(self.CFGForwardTime, self.CFGReverseTime, self.PFGIteraTime, self.MLPLayer,self.EmbedDim,self.OutputDim, self.DropoutRate)
        
    def call(self, CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND):
        basicBlock1st = self.data1.call(LITERAL1ST, SEMANTIC1ST)
        cfg1st = tf.cast(CFG1ST, tf.float32)
        embed1st = self.embed.call(basicBlock1st,cfg1st)

        basicBlock2nd = self.data2.call(LITERAL2ND, SEMANTIC2ND)
        cfg2nd = tf.cast(CFG2ND, tf.float32)
        embed2nd = self.embed.call(basicBlock2nd,cfg2nd)

        cosin = tf.keras.losses.cosine_similarity(embed1st, embed2nd, axis=1)
        label = tf.divide(-tf.subtract(cosin,1),2)

        return label

class MyDataLayer(layers.Layer):
    def __init__(self, LITERAL, SEMANTIC, EMBED_DIM, DROPOUT_RATE):
        super(MyDataLayer, self).__init__()
        self.Literal = LITERAL
        self.Semantic = SEMANTIC
        self.EmbedDim = EMBED_DIM
        self.HalfDim = 32
        self.DropoutRate = DROPOUT_RATE

        self.DenseIn = layers.Dense(self.EmbedDim,activation='relu')
        self.DenseOut = layers.Dense(self.HalfDim,activation='relu')
        #self.LSTM = layers.LSTM(self.HalfDim, activation='tanh',return_sequences=True, return_state=True)
        self.GRU = layers.GRU(self.HalfDim, activation='tanh',return_sequences=True, return_state=True)
        #self.Dropout = layers.Dropout(self.DropoutRate)
    
    def call(self, LITERAL, SEMANTIC):
        mlpMid = self.DenseIn(LITERAL)
        literal = self.DenseOut(mlpMid)
        #sematic,fms,fcs = self.LSTM(SEMANTIC)
        sematic,fs = self.GRU(SEMANTIC)
        output = tf.concat([literal,sematic],-1)
        #output = self.Dropout(output)
        return output

class MyGraphLayer(layers.Layer):
    def __init__(self, CFG_FORWARD_TIME, CFG_REVERSE_TIME, PFG_ITERA_TIME, EMBED_LAYER, EMBED_DIM, OUTPUT_DIM, DROPOUT_RATE):
        super(MyGraphLayer, self).__init__()
        self.CFGForwardTime = CFG_FORWARD_TIME
        self.CFGReverseTime = CFG_REVERSE_TIME
        self.PFGIteraTime = PFG_ITERA_TIME
        self.EmbedLayer = EMBED_LAYER
        self.DropoutRate = DROPOUT_RATE

        self.mlp = []
        for _ in range(self.EmbedLayer):
            self.mlp.append(layers.Dense(EMBED_DIM,activation='relu'))

        #self.Dropout = layers.Dropout(self.DropoutRate)

        w_init = tf.random_normal_initializer()
        self.wOutput = tf.Variable(initial_value=w_init(shape=(EMBED_DIM, OUTPUT_DIM), dtype = tf.float32),trainable=True)
        
        b_init = tf.zeros_initializer()
        self.bOutput = tf.Variable(initial_value=b_init(shape=(OUTPUT_DIM,), dtype = tf.float32),trainable=True)
    
    def call(self, INPUT, CFG):
        #Embedding by cfg
        cfgForwardEmbed = tf.nn.relu(INPUT)
        for _ in range(self.CFGForwardTime):
            #Message convey
            transMid = tf.matmul(CFG, cfgForwardEmbed)
            #Weight calculation
            for dense in self.mlp:
                transMid = dense(transMid)
            #Adding and Nonlinearity
            cfgForwardEmbed = tf.nn.tanh(cfgForwardEmbed + transMid)
        '''
        #Embedding by reversed cfg
        CFGReverse = np.transpose(CFG,(0,2,1))
        cfgReverseEmbed = tf.nn.relu(INPUT)
        for _ in range(self.CFGReverseTime):
            #Message convey
            transMid = tf.matmul(CFGReverse, cfgReverseEmbed)
            #Weight calculation
            for dense in self.mlp:
                transMid = dense(transMid)
            #Adding and Nonlinearity
            cfgReverseEmbed = tf.nn.tanh(cfgReverseEmbed + transMid)

        dropOutput = self.Dropout(cfgForwardEmbed+cfgReverseEmbed)
        '''
        midOutput = tf.reduce_sum(cfgForwardEmbed, 1)
        output = tf.matmul(midOutput, self.wOutput) + self.bOutput
    
        return output
