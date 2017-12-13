import tensorflow as tf
import numpy as np
from LoadData.DataRecord import *
def run():
    img_file = '../data/ProcessedData/face_image.csv'
    label_file = '../data/ProcessedData/mouth_right_corner.csv'
    data = FetchingData(img_file, label_file)
    data.splitData(0.9)
    data.dimY=0
    trainModel(data)
class dataHandle:
    data=None
    target=None

def trainModel(data):
    #Load datasets.
    training_set = dataHandle()
    test_set = dataHandle()
    training_set.data, training_set.target = data.getXY(data.trainSize)
    test_set.data, test_set.target = data.getXY(data.testSize)
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[96 * 96])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.LinearRegressor(feature_columns =feature_columns,
                                           #hidden_units=[200,150,80],
                                            model_dir="../data/models/")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_set.data},
        y=training_set.target,
        num_epochs=None,
        shuffle=True)

    # Train model.
    trainInfo=classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":test_set.data},
        y=test_set.target,
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    testInfo = classifier.evaluate(input_fn=test_input_fn,name="Linear Model")

    print("trainInfo",trainInfo)
    print("testInfo",testInfo)
    # Classify two new flower samples.
    x=np.zeros(shape=(2,96*96),dtype=np.float32)
    for j in range(2) :
        for i in range(96 * 96):
            p=random.randint(0,256)
            x[j][i]=p
    new_samples = x
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))

    print("New Samples, Reggression Predictions",predictions)

if __name__ == '__main__':
    run()