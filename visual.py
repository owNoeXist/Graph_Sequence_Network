import json
import matplotlib.pyplot as plt
#from sklearn.metrics import auc, roc_curve, precision_recall_curve
#--------------------------------------Make result Visualized-----------------------------------
def DrawTop(FILE_PATH = None, RAW_JSON = None):
    #Obtain data
    if FILE_PATH != None:
        file=open(FILE_PATH,'r')
        serialNum = json.loads(file.readlines()[0])["SerialNum"]
        file.close()
    else:
        serialNum = RAW_JSON["SerialNum"]
    #Exact coordinates from data
    dataNum = len(serialNum)
    groupNum = int((dataNum-1)/10)+1
    number = [0]*groupNum
    for i in range(dataNum):
        number[int((serialNum[i]-1)/10)]+=1
    maxNum = 0
    totalNum = 0
    group = [0]*groupNum
    totalPercent = [0]*groupNum
    groupPercent = [0]*groupNum
    for i in range(groupNum):
        maxNum = max(maxNum,number[i])
        group[i] = (i+1)*10
        totalNum += number[i]
        totalPercent[i] = totalNum/dataNum
        groupPercent[i] = (i+1)/groupNum
    #Initial drawing paper
    plt.rcParams['figure.figsize'] = (12, 5)
    graph=plt.subplot(1,2,1)
    graph.set_title('Top Serial Number',fontsize=15)
    graph.set_xlabel('Group',fontsize=10)
    x_major_locator=plt.MultipleLocator(0.1)
    graph.xaxis.set_major_locator(x_major_locator)
    graph.set_xlim(0,1)
    graph.set_ylabel('Number', fontsize=10)
    graph.set_ylim(0,maxNum)
    graph.plot(groupPercent, number, color='red', linewidth=1, linestyle='-', label='')
    graph=plt.subplot(1,2,2)
    graph.set_title('Top Serial Number',fontsize=15)
    graph.set_xlabel('Group',fontsize=10)
    x_major_locator=plt.MultipleLocator(0.1)
    graph.xaxis.set_major_locator(x_major_locator)
    graph.set_xlim(0,1)
    graph.set_ylabel('Number', fontsize=10)
    y_major_locator=plt.MultipleLocator(0.1)
    graph.yaxis.set_major_locator(y_major_locator)
    graph.set_ylim(0,1)
    graph.plot(groupPercent, totalPercent, color='red', linewidth=1, linestyle='-', label='')
    plt.show()

def DrawTrainLine(FILE_PATH = None, RAW_JSON = None):
    #Obtain data
    if FILE_PATH != None:
        file=open(FILE_PATH,'r')
        trainRecord = json.loads(file.readlines()[0])
        file.close()
    else:
        trainRecord = RAW_JSON
    #Initial drawing paper
    plt.rcParams['figure.figsize'] = (12, 5)
	#Draw Loss Line
    graphAuc=plt.subplot(1,2,1)
    graphAuc.grid(True,axis='y')
    graphAuc.set_title('Loss Graph',fontsize=15)
    graphAuc.set_xlabel('Step',fontsize=10)
    graphAuc.set_xlim(0, len(trainRecord["TrainStep"])-1)
    graphAuc.set_ylabel('Loss', fontsize=10)
    graphAuc.set_ylim(0,1)
    graphAuc.plot(trainRecord["TrainStep"],trainRecord["TrainLoss"],color='red',linewidth=1,linestyle='-')
    #Draw Auc Line
    graphAuc=plt.subplot(1,2,2)
    graphAuc.grid(True,axis='y')
    graphAuc.set_title('Auc Graph', fontsize=15)
    graphAuc.set_xlabel('Step', fontsize=10)
    graphAuc.set_xlim(0, len(trainRecord["TrainStep"])-1)
    graphAuc.set_ylabel('Auc', fontsize=10)
    graphAuc.set_ylim(0.8,1)
    graphAuc.plot(trainRecord["TestStep"],trainRecord["TrainAuc"],color='red',linewidth=1,linestyle='-')
    graphAuc.plot(trainRecord["TestStep"],trainRecord["ValidAuc"],color='green',linewidth=1,linestyle='-')
    plt.show()

def DrawRoc(Result_PATH, RAW_JSON=None):
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

def DrawPR(Result_PATH, RAW_JSON=None):
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