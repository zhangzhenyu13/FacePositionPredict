from sklearn import tree
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
import numpy as np
from RegressionLearner.ModelDesign import *
import time
class TreeRegressor(ModelDesign):
    def __init__(self,data):
        ModelDesign.__init__(self,data)

    def train(self):
        data=self.data
        X,Y=data.getXY(data.trainSize)
        X=np.array(X,dtype=np.float32)
        Y=np.array(Y,dtype=np.float32)
        Y=np.reshape(Y,newshape=Y.size)
        print("running tree regressor for",self.name)
        t1=time.time()
        self.model=tree.DecisionTreeRegressor()
        self.model=self.model.fit(X,Y)
        t2=time.time()
        print("finished in",t2-t1,"s")
        X,Y=data.getTestData()
        Y1=self.model.predict(X)
        Y=np.reshape(Y,newshape=Y.size)
        loss=np.sqrt(np.sum(np.square(Y1-Y)))
        print("test RMSE=",loss)
    def predict(self,x):
        y=self.model.predict(x)
        Y=np.reshape(y,newshape=(y.size,1))
        Y=self.data.rescale(Y)
        return Y

class BoostingTreeRegressor(ModelDesign):
    def __init__(self,data):
        ModelDesign.__init__(self,data)

    def train(self):
        data=self.data
        X,Y=data.getXY(data.trainSize)
        X=np.array(X,dtype=np.float32)
        Y=np.array(Y,dtype=np.float32)
        Y=np.reshape(Y,newshape=Y.size)
        print("running tree regressor for",self.name)
        t1=time.time()
        self.model=AdaBoostRegressor(loss='square')
        self.model=self.model.fit(X,Y)
        t2=time.time()
        print("finished in",t2-t1,"s")
        X,Y=data.getTestData()
        Y1=self.model.predict(X)
        Y=np.reshape(Y,newshape=Y.size)
        loss=np.sqrt(np.sum(np.square(Y1-Y)))
        print("test RMSE=",loss)
    def predict(self,x):
        y=self.model.predict(x)
        Y=np.reshape(y,newshape=(y.size,1))
        Y=self.data.rescale(Y)
        return Y


