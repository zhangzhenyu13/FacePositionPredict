from LoadData.DataRecord import *
import tensorflow as tf
import time
from RegressionLearner.ModelDesign import *
def runForTuning():
    print("This is a basic simple convolution net from basic api!!!")
    img_file = '../data/ProcessedData/face_image.csv'
    label_file = '../data/ProcessedData/mouth_right_corner.csv'
    data = FetchingData(img_file, label_file)
    data.splitData(0.9)

    method={}
    method["1"]=NN
    method["2"]=ConvNet
    print("choice:1-neuralNet,2-convNet")

    choice = input()
    learner=method[choice](data)
    learner.train()

    x,y=data.getTestData()
    y1=learner.predict(x)
    y=data.rescale(y)
    for i in range(len(x)):
        print(y[i],y1[i])


#this is very simple implementation of multi-layer perceptrons regression
class NN(ModelDesign):
    def __init__(self,data):
        self.data=data
        self.model=None
        self.learnRate = 1e-4
        self.batchSize = 50
        self.iterationNum = 3000
        # picSize:xy
        self.pX = 96
        self.pY = 96

        # define the Graph
        # data feed
        self.x = tf.placeholder(tf.float32, [None, self.pX * self.pY])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob=tf.placeholder(tf.float32)
        self.sess=None
        pass
    def predict(self,x):
        result=self.sess.run(self.model,feed_dict={self.x:x,self.keep_prob:1.0})
        result=self.data.rescale(result)
        return result
    def train(self):
        data=self.data
    #Capacity: only for gray pics

        #define the Graph

        #3 hidden layers
        W1=tf.Variable(tf.truncated_normal(shape=[self.pX*self.pY,500],stddev=0.1))#init for weight and bias
        b1=tf.Variable(tf.constant(value=0.1,shape=[500]))
        W2=tf.Variable(tf.truncated_normal(shape=[500,1000],stddev=0.1))
        b2=tf.Variable(tf.zeros(1000))
        W3=tf.Variable(tf.truncated_normal(shape=[1000,500],stddev=0.1))
        b3=tf.Variable(tf.zeros(500))
        W4=tf.Variable(tf.truncated_normal(shape=[500,1],stddev=0.1))
        b4=tf.Variable(tf.zeros(1))
        #define the hidden layer
        h1=tf.nn.relu(tf.matmul(self.x,W1)+b1)
        h2=tf.nn.relu(tf.matmul(h1,W2)+b2)
        h3=tf.nn.relu(tf.matmul(h2,W3)+b3)
        # define dropout
        h_prev=h3
        h_drop = tf.nn.dropout(h_prev, self.keep_prob)
        #define the output layer
        self.model=(tf.matmul(h_drop,W4)+b4)


        loss=tf.sqrt(tf.reduce_mean(tf.square(self.model-self.y)))
        train_unit=tf.train.AdagradOptimizer(learning_rate=self.learnRate).minimize(loss)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)


        #begin train
        print("running multi-layer perceptrons")
        t1=time.time()
        for i in range(1,self.iterationNum+1):
            batchX,batchC=data.getXY(self.batchSize)
            self.sess.run(train_unit,feed_dict={self.x:batchX,self.y:batchC,self.keep_prob:0.5})
            if i%100==0:
                print("step",i,"training RMSE=",self.sess.run(loss,feed_dict={self.x:batchX,self.y:batchC,self.keep_prob:1.0}))
                tX,tY=data.getTestData()
                print("step",i,"testing RMSE=",self.sess.run(loss,feed_dict={self.x:tX,self.y:tY,self.keep_prob:1.0}))
            #finish tensor replace by using part of data improves speed
        t2=time.time()
        print("training finished in",t2-t1,"s")


#this is a simple implementation of convolutional networks classifier
#part of the code is from tensorflow Tutorial
class ConvNet(ModelDesign):
    #class method
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    def max_pool_4x4(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 4, 4, 1], padding='SAME')
    def normalize(x):
        return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    #object method
    def __init__(self,data):
        self.data=data
        self.model=None
        self.predictDim=1

        self.learnRate = 1e-4
        self.batchSize = 50
        self.iterationNum = 3000
        # picSize:xy
        self.pX = 96
        self.pY = 96
        # define dataGraph
        self.x = tf.placeholder(tf.float32, [None, self.pX * self.pY])
        self.y = tf.placeholder(tf.float32, [None, 1])
        self.keep_prob=tf.placeholder(tf.float32)
        self.sess=None
    def predict(self,x):

        result=self.sess.run(self.model,feed_dict={self.x:x,self.keep_prob:1.0})
        result=self.data.rescale(result)

        return result

    def train(self):
        data=self.data
        #capacity: only for gray pics


        #define dataGraph

        x_image = tf.reshape(self.x, [-1, self.pX, self.pY, 1])

        W_conv1 = ConvNet.weight_variable([5, 5, 1, 32])#conv layer1 kernel 5*5
        b_conv1 = ConvNet.bias_variable([32])
        W_conv2 = ConvNet.weight_variable([5, 5, 32, 64])#conv layer2 kernel 5*5
        b_conv2 = ConvNet.bias_variable([64])
        W_conv3 = ConvNet.weight_variable([5,5,64,128])
        b_conv3 = ConvNet.bias_variable([128])
        W_fc1 = ConvNet.weight_variable([6*6 * 128, 1024])#total connect from 2dArray to hidden layer perceptrons(1024)
        b_fc1 = ConvNet.bias_variable([1024])
        W_fc2 = ConvNet.weight_variable([1024, 1])#connect from hiddenlayer to output layer(10)
        b_fc2 = ConvNet.bias_variable([1])

        # Create the 1st conv layer, reduce to 24*24,32
        h_conv1 = tf.nn.relu(ConvNet.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = ConvNet.max_pool_4x4(h_conv1)
        # create the 2nd conv, layer reduce to 12*12,64
        h_conv2 = tf.nn.relu(ConvNet.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = ConvNet.max_pool_2x2(h_conv2)
        #create the 3rd conv,layer reduce to 6*6,128
        h_conv3=tf.nn.relu(ConvNet.conv2d(h_pool2,W_conv3)+b_conv3)
        h_pool3=ConvNet.max_pool_2x2(h_conv3)
        # conv to multi-perceptrons layer, connect 6*6*64
        h_pool3_flat = tf.reshape(h_pool3, [-1, 6*6 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        # define dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # define output layer
        self.model=(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        # Define loss and optimizer

        loss = tf.sqrt(tf.reduce_mean(tf.square((self.y-self.model))))
        train_step = tf.train.AdamOptimizer(self.learnRate).minimize(loss)

        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #show data


        # Train
        print("runing convolutional networks")
        t1 = time.time()
        for i in range(1, self.iterationNum+1):
            batchX, batchY =data.getXY(self.batchSize)
            if i % 100 == 0:
                train_accuracy = loss.eval(feed_dict={
                    self.x: batchX, self.y: batchY, self.keep_prob: 1.0})
                print("step %d, training RMSE= %g" % (i, train_accuracy))
                tX,tY=data.getTestData()
                test_accuracy = loss.eval(feed_dict={
                    self.x: tX, self.y: tY, self.keep_prob: 1.0})
                print("step %d, testing RMSE= %g" % (i, test_accuracy))
                #print(sess.run(y_conv,feed_dict={x:batchX,y:batchY,keep_prob:1.0}))
            self.sess.run(train_step, feed_dict={self.x: batchX, self.y: batchY, self.keep_prob: 0.5})
        t2 = time.time()
        print("training finished in ",str(t2 - t1) + "s")


if __name__=="__main__":
    runForTuning()
