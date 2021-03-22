import matplotlib.pyplot as plt
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