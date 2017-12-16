import tensorflow as tf
import numpy as np
from queue import *
import threading
from RegressionLearner.ModelDesign import *
import os
from LoadData.DataRecord import *

def savePredict(file,result):
    print("writing result into file",file)

    with open(file,"w",newline="") as f:
        writer=csv.writer(f)
        for row in result:
            #print('type=',type(row))
            writer.writerow(row)
'''
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
            data.splitData(0.9)
            self.dataQueue.put(data)
            count=count+1

        self.getting=False
        print("getting finished ------> produced",count,"data")

    def getData(self):
        return self.dataQueue.get()

def run():
    label_file = '../data/ProcessedData/'
    getter=runPipe(label_file)
    getter.start()

    modelSet={}
    testData=TestData()
    files=os.listdir(label_file)
    count=1
    while getter.getting or getter.dataQueue.qsize()>0:
        print("#",count)
        count=count+1
        data=getter.getData()

        data.dimY=0
        learner=SimpleNN(data)
        learner.name = data.name + '_x'
        learner.train()
        #result=learner.predict(data.getTestData()[0])
        #save_file='../data/PredictResult/'+data.name+'_x.csv'
        #savePredict(save_file,result)          #save func,io very slow, be careful

        modelSet[learner.name]=learner

        data.dimY=1
        learner=SimpleNN(data)
        learner.name = data.name + '_y'
        learner.train()
        #result=learner.predict(data.getTestData()[0])
        #save_file='../data/PredictResult/'+data.name+'_y.csv'
        #savePredict(save_file,result)          #saved func, io too slow, be careful

        modelSet[learner.name]=learner

    testData.addModels(modelSet)
    testData.predictResults()
'''

class dataHandle:
    data=None
    target=None
class SimpleNN(ModelDesign):
    def __init__(self,data):
        ModelDesign.__init__(self,data)
        self.learningRate=1e-4
        self.iterNum=10000
    def predict(self,x):
        #x must be an array type
        #print(len(x),x)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x},
            num_epochs=1,
            shuffle=False)
        Y=[]
        predictions = list(self.model.predict(input_fn=predict_input_fn))
        for r in predictions:
            Y.append(r["predictions"])
        Y=self.data.rescale(Y)
        #print(Y)
        return Y

    def train(self):
        #Load datasets.
        data=self.data
        training_set = dataHandle()
        test_set = dataHandle()
        training_set.data, training_set.target = data.getXY(data.trainSize)
        test_set.data, test_set.target = data.getTestData()
        # Specify that all features have real-value data
        feature_columns = [tf.feature_column.numeric_column("x", shape=[96 * 96])]

        #Define regressor
        self.model = tf.estimator.LinearRegressor(feature_columns =feature_columns,
                                               #hidden_units=[200,150,80],
                                                model_dir="../data/models/")
        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": training_set.data},
            y=training_set.target,
            num_epochs=None,
            shuffle=True)

        # Train model.
        trainInfo=self.model.train(input_fn=train_input_fn, steps=self.iterNum)

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":test_set.data},
            y=test_set.target,
            num_epochs=1,
            shuffle=False)

        # Evaluate accuracy.
        testInfo = self.model.evaluate(input_fn=test_input_fn,name=self.name)

        print("model for ",self.name)
        print("trainInfo",trainInfo)
        print("testInfo",testInfo)


if __name__ == '__main__':
    pass
