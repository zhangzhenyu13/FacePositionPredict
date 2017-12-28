import threading
from queue import *
from LoadData.DataRecord import *
import os
import RegressionLearner.simpleDNN as snnreg
import RegressionLearner.CustomizedNN as cnnreg
import RegressionLearner.TraditionalRegressor as treg

#this class runs in back to save time for training
#this class produce data object and save them in a queue
class runPipe(threading.Thread):
    def __init__(self,filePath):
        threading.Thread.__init__(self)
        self.filePath=filePath
        self.dataQueue=Queue()
        self.getting=True
    def run(self):
        files=os.listdir(self.filePath)
        img_file = '../data/ProcessedData/face_image.csv'
        count=0
        for file in files:
            if file == 'face_image.csv':
                continue
            lf = self.filePath + file
            data = FetchingData(img_file, lf)
            #data.splitData(0.9)
            self.dataQueue.put(data)
            count=count+1

        self.getting=False
        print("getting finished ------> produced",count,"data")

    def getData(self):
        return self.dataQueue.get()

#given a model class, this function run all the train and save test predictions
def runEntry(model):
    label_file = '../data/ProcessedData/'
    getter=runPipe(label_file)
    getter.start()

    modelSet={}

    count=1
    while getter.getting or getter.dataQueue.qsize()>0:
        print("#",count)
        count=count+1
        data=getter.getData()

        data.dimY=0
        learner=model(data)
        learner.name = data.name + '_x'
        learner.train()
        #result=learner.predict(data.getTestData()[0])
        #save_file='../data/PredictResult/'+data.name+'_x.csv'
        #savePredict(save_file,result)          #save func,io very slow, be careful

        modelSet[learner.name]=learner

        data.dimY=1
        learner=model(data)
        learner.name = data.name + '_y'
        learner.train()
        #result=learner.predict(data.getTestData()[0])
        #save_file='../data/PredictResult/'+data.name+'_y.csv'
        #savePredict(save_file,result)          #saved func, io too slow, be careful

        modelSet[learner.name]=learner

    testData = TestData()
    testData.addModels(modelSet)
    testData.predictResults()


if __name__ == '__main__':
    #runEntry(snnreg.SimpleNN)
    #runEntry(cnnreg.ConvNet)

    runEntry(treg.MyRegressor)