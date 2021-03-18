import os
import json
from datetime import datetime

def TrainModel(MODEL, TRAIN_DATA, VALID_DATA, MODEL_PATH, TRAIN_EPOCH, TEST_FREQ, SAVE_FREQ):
    savePath=os.path.join(MODEL_PATH, MODEL.name())
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    savePath=os.path.join(savePath, MODEL.name())
    #Test model
    print("1")
    trainRocAuc = MODEL.TestModel(TRAIN_DATA)
    MODEL.say("Initial training auc = {0} @ {1}".format(trainRocAuc, datetime.now()))
    validRocAuc = MODEL.TestModel(VALID_DATA)
    MODEL.say("Initial validate auc = {0} @ {1}".format(validRocAuc, datetime.now()))

    trainStep = [0]
    trainLoss= [2]
    testStep = [0]
    trainAuc = [trainRocAuc]
    validAuc = [validRocAuc]

    bestAuc=0
    for i in range(1, TRAIN_EPOCH+1):
        MODEL.say("EPOCH {0}/{1} @ {2}".format(i, TRAIN_EPOCH, datetime.now()))
		#train for one epoch
        loss = MODEL.TrainModel(TRAIN_DATA)
        MODEL.say("Training loss= {0}".format(loss))
        trainStep.append(i)
        trainLoss.append(loss)

        #test model
        if (i % TEST_FREQ == 0):  
            trainRocAuc = MODEL.TestModel(TRAIN_DATA)
            MODEL.say("Training auc = {0}".format(trainRocAuc))

            validRocAuc = MODEL.TestModel(VALID_DATA)
            MODEL.say("Validate auc = {0}".format(validRocAuc))

            if validRocAuc > bestAuc+1 :
                path = MODEL.save(savePath+'_best')
                MODEL.say("Model saved in {}".format(path))
                bestAuc = validRocAuc

            testStep.append(i)
            trainAuc.append(trainRocAuc)
            validAuc.append(validRocAuc)

		
        if (i % SAVE_FREQ == 0):
            path = MODEL.save(savePath, i)
            MODEL.say("Model saved in {}".format(path))

    raw_json = {}
    raw_json["TrainStep"] = trainStep
    raw_json["TrainLoss"] = trainLoss
    raw_json["TestStep"] = testStep
    raw_json["TrainAuc"] = trainAuc
    raw_json["ValidAuc"] = validAuc
    filePath=os.path.join("./Result",(MODEL.name()+"-trainauc.txt"))
    file=open(filePath,'w')
    file.write(json.dumps(raw_json))
    file.write('\n')
    file.close()

    return raw_json

'''
#--------------------------------------Make result Visualized-----------------------------------
def DrawTrainLine(Result_PATH, RAW_JSON=None):
    if RAW_JSON == None:
        filePath=os.path.join(Result_PATH,"trainauc.txt")
        file=open(filePath,'r')
        graphInfo = json.loads(file.readlines().strip())
        file.close()
    else:
        graphInfo = RAW_JSON
    #Initial drawing paper
    plt.rcParams['figure.figsize'] = (12, 5)
	#Draw Loss Line
    graphAuc=plt.subplot(1,2,1)
    graphAuc.grid(True,axis='y')
    graphAuc.set_title('Loss Graph',fontsize=15)
    graphAuc.set_xlabel('Step',fontsize=10)
    graphAuc.set_xlim(0, len(graphInfo["TrainStep"])-1)
    graphAuc.set_ylabel('Loss', fontsize=10)
    graphAuc.set_ylim(0,1)
    graphAuc.plot(graphInfo["TrainStep"],graphInfo["TrainLoss"],color='red',linewidth=1,linestyle='-')
    #Draw Auc Line
    graphAuc=plt.subplot(1,2,2)
    graphAuc.grid(True,axis='y')
    graphAuc.set_title('Auc Graph', fontsize=15)
    graphAuc.set_xlabel('Step', fontsize=10)
    graphAuc.set_xlim(0, len(graphInfo["TrainStep"])-1)
    graphAuc.set_ylabel('Auc', fontsize=10)
    graphAuc.set_ylim(0.8,1)
    graphAuc.plot(graphInfo["TestStep"],graphInfo["TrainAuc"],color='red',linewidth=1,linestyle='-')
    graphAuc.plot(graphInfo["TestStep"],graphInfo["ValidAuc"],color='green',linewidth=1,linestyle='-')
    plt.show()

def Draw_Roc(Result_PATH, RAW_JSON=None):
    if RAW_JSON == None:
        filePath=os.path.join(Result_PATH,"roc.txt")
        file=open(filePath,'r')
        graphInfo = json.loads(file.readlines().strip())
        file.close()
    else:
        graphInfo = RAW_JSON
    #Initial drawing paper
    plt.rcParams['figure.figsize'] = (5, 5)
    graphRoc=plt.subplot(1,1,1)
    graphRoc.set_title('Roc Graph',fontsize=15)
    graphRoc.set_xlabel('False Positive Rate',fontsize=10)
    graphRoc.set_xlim(-0.05,1)
    graphRoc.set_ylabel('True Positive Rate', fontsize=10)
    graphRoc.set_ylim(0,1.05)
    graphRoc.plot(graphInfo["Fpr"], graphInfo["Tpr"], color='red', linewidth=1, linestyle='-', 
                    label='ROC curve (area = %0.3f)' % graphInfo["Auc"])
    graphRoc.legend(loc='lower right')
    plt.show()

def Draw_PR(Result_PATH, RAW_JSON=None):
    if RAW_JSON == None:
        filePath=os.path.join(Result_PATH,"pr.txt")
        file=open(filePath,'r')
        graphInfo = json.loads(file.readlines().strip())
        file.close()
    else:
        graphInfo = RAW_JSON
    #Initial drawing paper
    plt.rcParams['figure.figsize'] = (5, 5)
    graphRoc=plt.subplot(1,1,1)
    graphRoc.set_title('Precision-Recall Graph',fontsize=15)
    graphRoc.set_xlabel('Recall',fontsize=10)
    graphRoc.set_xlim(-0.05,1)
    graphRoc.set_ylabel('Precision', fontsize=10)
    graphRoc.set_ylim(0,1.05)
    graphRoc.plot(graphInfo["Recall"], graphInfo["Precision"], color='red', linewidth=1, linestyle='-', 
                    label='ROC curve (area = %0.3f)' % graphInfo["Auc"])
    graphRoc.legend(loc='lower right')
    plt.show()
'''