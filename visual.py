import os
import json
import matplotlib.pyplot as plt

LINE_COLOR = ['b','g','r','c','m','y','k','w']
LINE_STYLE = ['-','--','-.',':']
#--------------------------------------Make result Visualized-----------------------------------
def DrawResult(FOLDER,MODEL):
    plt.rcParams['figure.figsize'] = (12, 12)
    trainFile = []
    rocFile = []
    topFile = []
    for f in FOLDER:
        for m in MODEL:
            trainFile.append(os.path.join(f,m,'TrainLossAuc.json'))
            rocFile.append(os.path.join(f,m,'RocAuc.json'))
            topFile.append(os.path.join(f,m,'TopSerial.json'))
    DrawLoss(FILE_PATH=trainFile)
    DrawAuc(FILE_PATH=trainFile)
    DrawRoc(FILE_PATH=rocFile)
    DrawTop(FILE_PATH=topFile,PERCENT=True)
    plt.show()

def DrawLoss(FILE_PATH = None, RAW_JSON=None, TRAIN=True, VALID=True):
    #Initial drawing paper
    graphLoss=plt.subplot(2,2,1)
    graphLoss.grid(True,axis='y')
    graphLoss.set_title('Loss Graph',fontsize=12)
    graphLoss.set_xlabel('Step',fontsize=10)
    graphLoss.set_xlim(0, 50)
    graphLoss.set_ylabel('Loss', fontsize=10)
    graphLoss.set_ylim(0,1)
    #Load Result
    if FILE_PATH == None:
        resultTrain = RAW_JSON
        if TRAIN==True:
            graphLoss.plot(resultTrain["TrainStep"], resultTrain["TrainLoss"], color='b', 
                linewidth=1, linestyle='-', label='c')
        if VALID==True:
            graphLoss.plot(resultTrain["TrainStep"], resultTrain["ValidLoss"], color='m', 
                linewidth=1, linestyle='-', label='Valid')
    else:
        fileNum = len(FILE_PATH)
        for i in range(fileNum):
            file=open(FILE_PATH[i],'r')
            resultTrain = json.loads(file.readlines()[0])
            file.close()
            left=FILE_PATH[i].find('\\')
            right=FILE_PATH[i].find('\\',left+1)
            modelName=FILE_PATH[i][left+1:right]
            if TRAIN==True:
                graphLoss.plot(resultTrain["TrainStep"], resultTrain["TrainLoss"], color=LINE_COLOR[i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{}-Train'.format(modelName))
            if VALID==True:
                graphLoss.plot(resultTrain["TrainStep"], resultTrain["ValidLoss"], color=LINE_COLOR[4+i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{}-Valid'.format(modelName))
    graphLoss.legend(loc='lower right')

def DrawAuc(FILE_PATH = None, RAW_JSON=None, TRAIN=True, VALID=True):
    #Initial drawing paper
    graphAuc=plt.subplot(2,2,2)
    graphAuc.grid(True,axis='y')
    graphAuc.set_title('Auc Graph', fontsize=12)
    graphAuc.set_xlabel('Step', fontsize=10)
    graphAuc.set_xlim(0, 50)
    graphAuc.set_ylabel('Auc', fontsize=10)
    graphAuc.set_ylim(0.8,1)
    #Load Result
    if FILE_PATH == None:
        resultTrain = RAW_JSON
        if TRAIN==True:
            graphAuc.plot(resultTrain["TestStep"], resultTrain["TrainAuc"], color='b', 
                linewidth=1, linestyle='-', label='Valid')
        if VALID==True:
            graphAuc.plot(resultTrain["TestStep"], resultTrain["ValidAuc"], color='m', 
                linewidth=1, linestyle='-', label='Valid')
    else:
        fileNum = len(FILE_PATH)
        for i in range(fileNum):
            file=open(FILE_PATH[i],'r')
            resultTrain = json.loads(file.readlines()[0])
            file.close()
            left=FILE_PATH[i].find('\\')
            right=FILE_PATH[i].find('\\',left+1)
            modelName=FILE_PATH[i][left+1:right]
            if TRAIN==True:
                graphAuc.plot(resultTrain["TestStep"], resultTrain["TrainAuc"], color=LINE_COLOR[i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{}-Train'.format(modelName))
            if VALID==True:
                graphAuc.plot(resultTrain["TestStep"], resultTrain["ValidAuc"], color=LINE_COLOR[4+i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{}-Valid'.format(modelName))
    graphAuc.legend(loc='lower right')

def DrawRoc(FILE_PATH = None, RAW_JSON=None, TRAIN=True, VALID=True):
    #Initial drawing paper
    graphRoc=plt.subplot(2,2,3)
    graphRoc.set_title('Roc Graph',fontsize=12)
    graphRoc.set_xlabel('False Positive Rate',fontsize=10)
    graphRoc.set_xlim(-0.01,1)
    graphRoc.set_ylabel('True Positive Rate', fontsize=10)
    graphRoc.set_ylim(0,1.01)
    #Load Result
    if FILE_PATH == None:
        resultRoc = RAW_JSON
        if TRAIN==True:
            graphRoc.plot(resultRoc["TrainFpr"], resultRoc["TrainTpr"], color='b', 
                linewidth=1, linestyle='-', label='Train ROC (AUC={:.3f})'.format(resultRoc["TrainAuc"]))
        if VALID==True:
            graphRoc.plot(resultRoc["ValidFpr"], resultRoc["ValidTpr"], color='m', 
                linewidth=1, linestyle='-', label='Valid ROC (AUC={:.3f})'.format(resultRoc["ValidAuc"]))
    else:
        fileNum = len(FILE_PATH)
        for i in range(fileNum):
            file=open(FILE_PATH[i],'r')
            resultRoc = json.loads(file.readlines()[0])
            file.close()
            left=FILE_PATH[i].find('\\')
            right=FILE_PATH[i].find('\\',left+1)
            modelName=FILE_PATH[i][left+1:right]
            if TRAIN==True:
                graphRoc.plot(resultRoc["TrainFpr"], resultRoc["TrainTpr"], color=LINE_COLOR[i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{} Train (AUC = {:.3f})'.format(modelName,resultRoc["TrainAuc"]))
            if VALID==True:
                graphRoc.plot(resultRoc["ValidFpr"], resultRoc["ValidTpr"], color=LINE_COLOR[4+i], 
                    linewidth=1, linestyle=LINE_STYLE[i], label='{} Valid (AUC = {:.3f})'.format(modelName,resultRoc["ValidAuc"]))
    graphRoc.legend(loc='lower right')

def DrawTop(FILE_PATH = None, RAW_JSON = None, PERCENT = False):
    #Initial drawing paper
    graphTop=plt.subplot(2,2,4)
    if PERCENT==False:
        graphTop.set_title('Top Serial Number',fontsize=12)
        graphTop.set_xlabel('Group Percent',fontsize=10)
        graphTop.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        graphTop.set_xlim(0,0.1)
        graphTop.set_ylabel('Number', fontsize=10)
        graphTop.set_ylim(0,500)
    else:    
        graphTop.set_title('Top Serial Number',fontsize=12)
        graphTop.set_xlabel('Group Percent',fontsize=10)
        graphTop.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        graphTop.set_xlim(0,0.1)
        graphTop.set_ylabel('Number Percent', fontsize=10)
        graphTop.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        graphTop.set_ylim(0,1)
    #Load Result
    if FILE_PATH == None:
        resultTop = RAW_JSON
        group,total = Smooth(resultTop['SerialNum'],PERCENT)
        graphTop.plot(group, total, color='red', linewidth=1, 
            linestyle='-', label='None')
    else:
        fileNum = len(FILE_PATH)
        for i in range(fileNum):
            file=open(FILE_PATH[i],'r')
            resultTop = json.loads(file.readlines()[0])
            file.close()
            left=FILE_PATH[i].find('\\')
            right=FILE_PATH[i].find('\\',left+1)
            modelName=FILE_PATH[i][left+1:right]
            group,total = Smooth(resultTop['SerialNum'],PERCENT)
            graphTop.plot(group, total, color=LINE_COLOR[i], linewidth=1, linestyle=LINE_STYLE[i], 
                label='{}'.format(modelName))
    graphTop.legend(loc='lower right')

def Smooth(serialNum, PERCENT = False):
    #Exact coordinates from data
    dataNum = len(serialNum)
    groupNum = int((dataNum-1)/10)+1
    totalNumber = [0]*groupNum
    for i in range(dataNum):
        totalNumber[int((serialNum[i]-1)/10)]+=1
    maxNum = 0
    totalNum = 0
    group = [0]*groupNum
    totalPercent = [0]*groupNum
    groupPercent = [0]*groupNum
    for i in range(groupNum):
        maxNum = max(maxNum,totalNumber[i])
        group[i] = (i+1)*10
        totalNum += totalNumber[i]
        totalPercent[i] = totalNum/dataNum
        groupPercent[i] = (i+1)/groupNum
    if PERCENT == False:
        return groupPercent,totalNumber
    else:
        return groupPercent,totalPercent

if __name__ == '__main__':
    folderDir = ['C_X86-O0']
    modelType = ['GSN','Gemini']
    DrawResult(folderDir,modelType)