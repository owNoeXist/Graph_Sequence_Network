import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers,Model
from sklearn.metrics import auc, roc_curve, precision_recall_curve

class GSN():
    def __init__(self, HYPER_PARAMETER):
        self.model = MyModel(HYPER_PARAMETER)
        self.lossObject = tf.keras.losses.BinaryCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.trainLoss = tf.keras.metrics.Mean()
        self.logFile = None

    def name(self):
        return type(self).__name__

    def say(self, string):
        print(string)
        if self.logFile != None:
            self.logFile.write(string+'\n')

    def TrainModel(self, EPOCH_DATA):
        perm = np.random.permutation(len(EPOCH_DATA))
        cumLoss = 0.0

        for index in perm:
            c1, p1, l1 ,s1 ,c2, p2, l2, s2, y = EPOCH_DATA[index]
            loss = self.train_step(c1, p1, l1 ,s1 ,c2, p2, l2, s2, y)
            cumLoss += loss

        return cumLoss / len(perm)


    def train_step(self, CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND, LABELS):
        with tf.GradientTape() as tape:
            predictions = self.model(CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND)
            loss = self.lossObject(LABELS, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def TestModel(self, EPOCH_DATA, AUC_ONLY=True, PR_ROC=2):
        totCos = []
        totTruth = []

        for curData in EPOCH_DATA:
            c1, p1, l1 ,s1 ,c2, p2, l2, s2, y  = curData
            cos = self.model(c1, p1, l1 ,s1 ,c2, p2, l2, s2)
            totCos += list(cos)
            totTruth += list(y)
        predc = np.array(totCos)
        truth = np.array(totTruth)

        #Return result with different standard
        if PR_ROC%2 == 1:
            pre, rec, prThres = precision_recall_curve(truth, predc)
            prAuc = auc(rec, pre)
        if (PR_ROC/2)%2 == 1:
            fpr, tpr, rocThres = roc_curve(truth, predc)
            rocAuc = auc(fpr,tpr)
        if PR_ROC==0:
            return
        elif PR_ROC==1:
            if AUC_ONLY:
                return prAuc
            else:
                return prAuc, pre, rec, prThres
        elif PR_ROC==2:
            if AUC_ONLY:
                return rocAuc
            else:
                return rocAuc, fpr, tpr, rocThres
        elif PR_ROC==3:
            if AUC_ONLY:
                return rocAuc, prAuc
            else:
                return rocAuc, fpr, tpr, rocThres, prAuc, pre, rec, prThres

    @tf.function
    def test_step(self, CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND, LABELS):
        predictions = self.model(CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND)
        loss = self.lossObject(LABELS, predictions)

        return loss

class MyModel(Model):
    def __init__(self, HYPER_PARAMETER):
        self.Literal1st = HYPER_PARAMETER['Literal1st']
        self.Semantic1st = HYPER_PARAMETER['Semantic1st']
        self.Literal2nd = HYPER_PARAMETER['Literal2nd']
        self.Semantic2nd = HYPER_PARAMETER['Semantic2nd']
        self.IteraTimes = HYPER_PARAMETER['IteraTimes']
        self.EmbedLayer = HYPER_PARAMETER['EmbedLayer']
        self.EmbedDim = HYPER_PARAMETER['EmbedDim']
        self.OutputDim = HYPER_PARAMETER['OutputDim']
        self.BATCH_SIZE = HYPER_PARAMETER['Multiple']*HYPER_PARAMETER['BatchSize']
        
        super(MyModel, self).__init__()
        self.data1 = MyDataLayer(self.BATCH_SIZE, self.Literal1st, self.Semantic1st, self.EmbedDim)
        self.data2 = MyDataLayer(self.BATCH_SIZE, self.Literal2nd, self.Semantic2nd, self.EmbedDim)
        self.embed = MyEmbedLayer(self.IteraTimes, self.EmbedLayer,self.EmbedDim,self.OutputDim)
        
    def call(self, CFG1ST, PFG1ST, LITERAL1ST, SEMANTIC1ST, CFG2ND, PFG2ND, LITERAL2ND, SEMANTIC2ND):
        basicBlock1st = self.data1.call(LITERAL1ST, SEMANTIC1ST)
        cfg1st = tf.cast(CFG1ST, tf.float32)
        embed1st = self.embed.call(basicBlock1st,cfg1st)

        basicBlock2nd = self.data2.call(LITERAL2ND, SEMANTIC2ND)
        cfg2nd = tf.cast(CFG2ND, tf.float32)
        embed2nd = self.embed.call(basicBlock2nd,cfg2nd)

        cos = tf.reduce_sum(embed1st*embed2nd, 1) / tf.sqrt(
            tf.reduce_sum(embed1st**2, 1) * tf.reduce_sum(embed2nd**2, 1) + 1e-10)

        return cos

class MyDataLayer(layers.Layer):
    def __init__(self, BATCH_SIZE, LITERAL, SEMANTIC, EMBED_DIM):
        super(MyDataLayer, self).__init__()
        self.BatchSize = BATCH_SIZE
        self.Literal = LITERAL
        self.Semantic = SEMANTIC
        self.EmbedDim = EMBED_DIM
        self.HalfDim = 32
        self.Dense1 = layers.Dense(self.EmbedDim, activation='relu')
        self.Dense2 = layers.Dense(self.HalfDim, activation='relu')
        self.lstm = layers.LSTM(self.HalfDim, activation='tanh',return_sequences=True, return_state=True)
    
    def call(self, LITERAL, SEMANTIC):
        #MLP
        mlpMid = self.Dense1(LITERAL)
        mlpEmbed = self.Dense2(mlpMid)
        #LSTM
        lstmEmbed,fms,fcs = self.lstm(SEMANTIC)
        #Merge
        output = tf.concat([mlpEmbed,lstmEmbed],-1)
        return output

class MyEmbedLayer(layers.Layer):
    def __init__(self, ITERA_TIMES, EMBED_LAYER, EMBED_DIM, OUTPUT_DIM):
        super(MyEmbedLayer, self).__init__()

        self.IteraTimes = ITERA_TIMES
        self.EmbedLayer = EMBED_LAYER
        self.wEmbed = []
        for _ in range(self.EmbedLayer):
            self.wEmbed.append(layers.Dense(EMBED_DIM,activation='relu'))

        w_init = tf.random_normal_initializer()
        self.wOutput = tf.Variable(initial_value=w_init(shape=(EMBED_DIM, OUTPUT_DIM), dtype = tf.float32),trainable=True)
        
        b_init = tf.zeros_initializer()
        self.bOutput = tf.Variable(initial_value=b_init(shape=(OUTPUT_DIM,), dtype = tf.float32),trainable=True)
    
    def call(self, INPUT, NODE_MASK):
        xInput = tf.nn.relu(INPUT)
        for _ in range(self.IteraTimes):
            #Message convey
            transMid = tf.matmul(NODE_MASK, xInput)
            #Weight calculation
            for embed in self.wEmbed:
                transMid = embed(transMid)
            #Adding and Nonlinearity
            xInput = tf.nn.tanh(INPUT + transMid)
    
        xOutput = tf.reduce_sum(xInput, 1)
        output = tf.matmul(xOutput, self.wOutput) + self.bOutput
    
        return output