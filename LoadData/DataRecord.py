import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import queue
import sklearn.preprocessing as preData

def map1(data):
    out=[]
    try:
        out=list(map(float,data[:30]))
    except:
        for i in range(len(data)):
            if data[i]=='':
                out.append(None)
            else:
                out.append(eval(data[i]))
    return out
class DataRecord():

    def __init__(self):

        print("init data Set ...")
        f=open('../data/OriginalData/training.csv',"r")
        reader=csv.reader(f)
        #define the position data as key points
        self.positionData={}
        self.left_eye_center={}
        self.right_eye_center={}
        self.left_eye_innner_corner={}
        self.right_eye_inner_corner={}
        self.left_eye_outer_corner={}
        self.right_eye_outer_corner={}
        self.left_eyebrow_outer_end={}
        self.right_eyebrow_outer_end={}
        self.left_eyebrow_inner_end={}
        self.right_eyebrow_inner_end={}
        self.nose_tip={}
        self.mouth_left_corner={}
        self.mouth_right_corner={}
        self.mouth_center_top_lip={}
        self.mouth_center_bottom_lip={}
        'add position data to the root dict'
        self.positionData['left_eye_center']=self.left_eye_center
        self.positionData['right_eye_center']=self.right_eye_center
        self.positionData['left_eye_inner_corner']=self.left_eye_innner_corner
        self.positionData['left_eye_outer_corner']=self.left_eye_outer_corner
        self.positionData['right_eye_inner_corner']=self.right_eye_inner_corner
        self.positionData['right_eye_outer_corner']=self.right_eye_outer_corner
        self.positionData['left_eyebrow_inner_end']=self.left_eyebrow_inner_end
        self.positionData['left_eyebrow_outer_end']=self.left_eyebrow_outer_end
        self.positionData['right_eyebrow_inner_end']=self.right_eyebrow_inner_end
        self.positionData['right_eyebrow_outer_end']=self.right_eyebrow_outer_end
        self.positionData['nose_tip']=self.nose_tip
        self.positionData['mouth_left_corner']=self.mouth_left_corner
        self.positionData['mouth_right_corner']=self.mouth_right_corner
        self.positionData['mouth_center_top_lip']=self.mouth_center_top_lip
        self.positionData['mouth_center_bottom_lip']=self.mouth_center_bottom_lip
        #image data
        self.face_image={}


        #load data from csv file
        countID=0
        header=True
        for row in reader:
            #print(row)

            if header:
                header=False
                continue
            #data id start from 1
            countID = countID + 1
            # load image
            self.face_image[countID] = tuple(map(int, row[30].split(" ")))
            #laod labels
            row=map1(row[:30])
            self.left_eye_center[countID]=((row[0]),(row[1]))
            self.right_eye_center[countID]=((row[2]),(row[3]))
            self.left_eye_innner_corner[countID]=((row[4]),(row[5]))
            self.left_eye_outer_corner[countID] = ((row[6]), (row[7]))
            self.right_eye_inner_corner[countID]=((row[8]),(row[9]))
            self.right_eye_outer_corner[countID]=((row[10]),(row[11]))
            self.left_eyebrow_inner_end[countID]=((row[12]),(row[13]))
            self.left_eyebrow_outer_end[countID]=((row[14]),(row[15]))
            self.right_eyebrow_inner_end[countID]=((row[16]),(row[17]))
            self.right_eyebrow_outer_end[countID]=((row[18]),(row[19]))
            self.nose_tip[countID]=((row[20]),(row[21]))
            self.mouth_left_corner[countID]=((row[22]),(row[23]))
            self.mouth_right_corner[countID]=((row[24]),(row[25]))
            self.mouth_center_top_lip[countID]=((row[26]),(row[27]))
            self.mouth_center_bottom_lip[countID]=((row[28]),(row[29]))

        #finished
        self.Size=countID
        print("data Set Size=",self.Size)

    def countMissing(self):
        #print("Statistics in labels missing count")
        for k in self.positionData.keys():
            labels=self.positionData[k]
            count=0
            #print("Counting label=",k)
            labels["maintainID"]=set()
            for k1 in labels.keys():
                try:
                    if None in labels[k1] and k1!="maintainID":
                        count=count+1
                    else:
                        if k1!="maintainID":
                            labels["maintainID"].add(k1)
                except:
                    print("Error--->",k,k1,labels[k1])

            labels["missing"]=count
            #print(labels["maintainID"])
        #print("Statistics Counting finished")

    def dataDsitributionPlot(self):
        count = 1
        #plt.figure(count)
        for k in self.positionData.keys():
            labels = self.positionData[k]
            plt.figure(k)
            count = count + 1
            x = np.zeros(len(labels["maintainID"]))
            y = np.zeros(len(labels["maintainID"]))
            index = 0
            for i in range(1, self.Size + 1):
                if None not in labels[i]:
                    x[index], y[index] = labels[i]
                    index = index + 1
            plt.plot(x, y)
        plt.show()

class TestData:
    def __init__(self):
        self.reqTable=[]
        self.images={}
        with open("../data/OriginalData/test.csv","r") as f:
            reader=csv.reader(f)
            header=True
            for row in reader:
                if header:
                    header=False
                    continue
                self.images[row[0]]=list(map(int,row[1].split(" ")))
        with open("../data/OriginalData/IdLookupTable.csv","r") as f:
            reader=csv.reader(f)
            header=True
            for row in reader:
                if header:
                    header=False
                    continue
                self.reqTable.append(row)

    def addModels(self,modelList):
            self.models=modelList
    def predictResults(self):
        with open("../data/PredictResult/result.csv","w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(['RowId','Location'])
            for record in self.reqTable:
                result=self.models[record[2]].predict(self.images[record[1]])
                writer.writerow([record[0],result])


#pipeline fetching data
class FetchingData:
    def __init__(self,images_file,labels_file):

        # define a cache area for pipeline fetching data
        self.trainSize =None
        self.testSize = None
        self.testImg = []
        self.testLabel = []

        print("init data set")
        self.name=""
        self.scalarX=preData.MinMaxScaler()
        self.scalarY=preData.MinMaxScaler()
        self.labels={}
        self.images={}
        count = 0

        # load labels data
        self.dimY=1
        x=[]
        y=[]
        with open(labels_file, "r") as f:
            label_reader = csv.reader(f)
            header = True
            for row in label_reader:
                if header:
                    header = False
                    continue

                self.labels[row[0]] = [eval(row[1]), eval(row[2])]
                x.append([eval(row[1])])
                y.append([eval(row[2])])
                count = count + 1
        self.scalarX.fit(x)
        self.scalarY.fit(y)
        #load those image data with id(s) in the given labels
        with open(images_file,"r") as f:
            image_reader = csv.reader(f)
            header = True
            for row in image_reader:
                if header:
                    header = False
                    continue
                if row[0] not in self.labels.keys():
                    continue

                self.images[row[0]] = np.array(list(map(int,row[1].split(" "))))

        self.Size=count
        print("size=",self.Size)

    def rescale(self,y):
        dimY=self.dimY
        if dimY==0:
            y=self.scalarX.inverse_transform(y)
        elif dimY==1:
            y=self.scalarY.inverse_transform(y)
        return y
    def splitData(self,ratio):
        self.trainSize = int(self.Size * ratio)
        self.testSize = self.Size - self.trainSize
        indexL=list(self.labels.keys())
        random.shuffle(indexL)


        for i in range(self.testSize):
            k=indexL[i]
            self.testImg.append(self.images[k])
            self.images.pop(k)
            self.testLabel.append(self.labels[k])
            self.labels.pop(k)

        print("split ratio=",ratio,"testSize/trainSize=",self.testSize,self.trainSize)

    def getTestData(self):
        dimY=self.dimY
        X=[]
        Y=[]
        X=self.testImg
        Y=self.testLabel
        X = np.array(X)
        Y = np.array(Y)
        Y = Y[:, dimY:dimY + 1]

        if dimY==0:
            Y=self.scalarX.transform(Y)
        elif dimY==1:
            Y=self.scalarY.transform(Y)




        return X,Y
    def getXY(self,batchSize):
        #print("fetching")
        dimY=self.dimY
        indexL=list(self.labels.keys())
        random.shuffle(indexL)
        X=[]
        Y=[]
        for i in range(batchSize):
            X.append(self.images[indexL[i]])
            Y.append(self.labels[indexL[i]])

        X = np.array(X)
        Y = np.array(Y)
        Y = Y[:, dimY:dimY + 1]

        if dimY == 0:
            Y = self.scalarX.transform(Y)
        elif dimY == 1:
            Y = self.scalarY.transform(Y)

        return X,Y
#test

if __name__=="__main__":
    '''
    dataR=DataRecord()
    dataR.countMissing()
    for k in dataR.positionData.keys():
        print(k,"missing=",dataR.positionData[k]["missing"])
    dataR.dataDsitributionPlot()
    '''


    img_file='../data/ProcessedData/face_image.csv'
    label_file='../data/ProcessedData/mouth_right_corner.csv'
    data=FetchingData(img_file,label_file)
    data.splitData(0.9)

    for i in range(2):
        x,y=data.getXY(5)
        print("data x,y ")
        for j in range(len(x)):
            print(x[i])
            print(y[i])

