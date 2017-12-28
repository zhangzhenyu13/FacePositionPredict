from sklearn import tree
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn import svm
from RegressionLearner.ModelDesign import *
import time
from LoadData.DataRecord import *
def runForTuning():
    img_file = '../data/ProcessedData/face_image.csv'
    label_file = '../data/ProcessedData/mouth_right_corner.csv'
    data = FetchingData(img_file, label_file)
    data.splitData(0.9)

    learner=MyRegressor(data)
    method={}
    method["1"]=tree.ExtraTreeRegressor
    method["2"]=RandomForestRegressor
    method["3"]=AdaBoostRegressor
    method["4"]=svm.NuSVR


    choice = "2"
    learner.model=method[choice]()
    learner.train()

    x,y=data.getTestData()
    y1=learner.predict(x)
    for i in range(len(x)):
        print(y[i],y1[i])

class MyRegressor(ModelDesign):
    def __init__(self,data):
        ModelDesign.__init__(self,data)
        self.reduction=PCA(n_components=150)
        self.reduction=FactorAnalysis(n_components=150)
        self.model=svm.NuSVR()
    def train(self):
        data=self.data
        X,Y=data.getXY(data.trainSize)
        X=np.array(X,dtype=np.float32)
        Y=np.array(Y,dtype=np.float32)
        Y=np.reshape(Y,newshape=Y.size)
        print("running", self.model ,"regressor for",self.name)
        t1=time.time()
        self.reduction.fit(X)
        X=self.reduction.transform(X)
        self.model=self.model.fit(X,Y)
        t2=time.time()
        print("finished in",t2-t1,"s")
        X,Y=data.getTestData()
        if X is not None:
            X=self.reduction.transform(X)
            Y1=self.model.predict(X)
            Y=np.reshape(Y,newshape=Y.size)
            loss=np.sqrt(np.mean(np.square(Y1-Y)))
            print("test RMSE=",loss)
    def predict(self,x):
        x=self.reduction.transform(x)
        y=self.model.predict(x)
        Y=np.reshape(y,newshape=(y.size,1))
        #Y=self.data.rescale(Y)
        return Y



if __name__ == '__main__':
    runForTuning()