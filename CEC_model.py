import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers,Model
from sklearn.metrics import auc, roc_curve, precision_recall_curve

class CEC():
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

#================================================================================================

    def TrainModel(self, EPOCH_DATA):
        perm = np.random.permutation(len(EPOCH_DATA))   #Random shuffle
        cumLoss = 0.0

        for index in perm:
            x1, x2, m1, m2, y = EPOCH_DATA[index]
            loss = self.train_step(x1, m1, x2, m2, y)
            cumLoss += loss

        return cumLoss / len(perm)


    def train_step(self, INPUT1, NODE_MASK1, INPUT2, NODE_MASK2, LABELS):
        with tf.GradientTape() as tape:
            predictions = self.model(INPUT1, NODE_MASK1, INPUT2, NODE_MASK2)
            loss = self.lossObject(LABELS, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def TestModel(self, EPOCH_DATA, AUC_ONLY=True, PR_ROC=2):
        totCos = []
        totTruth = []

        for curData in EPOCH_DATA:
            x1, x2, m1, m2, y  = curData
            totTruth += list(y)
            cos = self.model(x1, m1, x2, m2)
            totCos += list(cos)
        truth = np.array(totTruth)
        predc = np.array(totCos)

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
    def test_step(self, INPUT1, NODE_MASK1, INPUT2, NODE_MASK2, LABELS):
        predictions = self.model(INPUT1, NODE_MASK1, INPUT2, NODE_MASK2)
        loss = self.lossObject(LABELS, predictions)

        return loss

class MyModel(Model):
    def __init__(self, HYPER_PARAMETER):
        self.Line1st = HYPER_PARAMETER['Line1st']
        self.Word1st = HYPER_PARAMETER['Word1st']
        self.Line2nd = HYPER_PARAMETER['Line2nd']
        self.Word2nd = HYPER_PARAMETER['Word2nd']
        self.IteraTimes = HYPER_PARAMETER['IteraTimes']
        self.EmbedLayer = HYPER_PARAMETER['EmbedLayer']
        self.EmbedDim = HYPER_PARAMETER['EmbedDim']
        self.OutputDim = HYPER_PARAMETER['OutputDim']
        self.BATCH_SIZE = 2*HYPER_PARAMETER['BatchSize']
        
        super(MyModel, self).__init__()
        self.data1 = MyDataLayer(self.BATCH_SIZE, self.Line1st, self.Word1st, self.EmbedDim)
        self.data2 = MyDataLayer(self.BATCH_SIZE, self.Line2nd, self.Word2nd, self.EmbedDim)
        self.embed = MyEmbedLayer(self.IteraTimes, self.EmbedLayer,self.EmbedDim,self.OutputDim)
        
    def call(self, INPUT1, NODE_MASK1, INPUT2, NODE_MASK2):
        input1_lstm = self.data1.call(INPUT1)
        node_mask1=tf.cast(NODE_MASK1, tf.float32)
        input1_embed = self.embed.call(input1_lstm,node_mask1)

        input2_lstm = self.data2.call(INPUT2)
        node_mask2=tf.cast(NODE_MASK2, tf.float32)
        input2_embed = self.embed.call(input2_lstm,node_mask2)

        cos = tf.reduce_sum(input1_embed*input2_embed, 1) / tf.sqrt(
            tf.reduce_sum(input1_embed**2, 1) * tf.reduce_sum(input2_embed**2, 1) + 1e-10)

        return cos

class MyDataLayer(layers.Layer):
    def __init__(self, BATCH_SIZE, LINE, WORD, EMBED_DIM):
        super(MyDataLayer, self).__init__()
        self.BatchSize = BATCH_SIZE
        self.Line = LINE
        self.Word = WORD
        self.EmbedDim = EMBED_DIM
        self.lstm = layers.LSTM(self.EmbedDim)
    
    def call(self, INPUT):
        rInput = tf.reshape(INPUT, [-1, self.Line, self.Word])
        rOutput = self.lstm(rInput)
        output = tf.reshape(rOutput, [self.BatchSize, -1, self.EmbedDim])
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