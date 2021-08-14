import numpy as np
import matplotlib.pyplot as plt
import time
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x/(np.sum(exp_x,axis=1,keepdims=True)+0.00001)

def compute(X, W, b):
    return np.matmul(W,X) + b

def averageCE(target, prediction):
    return -np.mean(target*np.log(prediction+0.00001))

def gradCE(target, prediction):
    return prediction-target

def grad_o_w(target, prediction, hidden_o):
    grad_ce = gradCE(target,prediction)
    return np.matmul(hidden_o.T,grad_ce)

def grad_o_b(target, prediction):
    grad_ce = gradCE(target,prediction)
    vector = np.ones((1,target.shape[0]))
    return np.matmul(vector,grad_ce)

def grad_h_w(target, prediction, X, Z_h, W_o):
    Z_h = np.where(Z_h>0,1,0)
    grad_ce = gradCE(target,prediction)
    return np.matmul(X,Z_h*np.matmul(grad_ce,W_o.T))

def grad_h_b(target, prediction, Z_h, W_o):
    Z_h = np.where(Z_h>0,1,0)
    grad_ce = gradCE(target,prediction)
    vector = np.ones((1,Z_h.shape[0]))
    return np.matmul(vector,Z_h*np.matmul(grad_ce,W_o.T))

def train(epochs, trainData, trainTarget, validData, validTarget, W_o, v_o, b_o, W_h, v_h, b_h, gamma, lr):
    accuracy_train = []
    accuracy_valid = []
    loss_train = []
    loss_valid = []
    v_out_weight = v_o
    v_hidden_weight = v_h
    v_out_bias = b_o
    v_hidden_bias = b_h
    for epoch in range(epochs):
        z_h = np.matmul(W_h.T,trainData).T + b_h
        h = relu(z_h)
        o = np.matmul(h,W_o) + b_o
        p = softmax(o)
        loss_t = averageCE(trainTarget, p)
        loss_train.append(loss_t)
        predictions = np.argmax(p,axis=1)
        real = np.argmax(trainTarget,axis=1)
        correct = np.equal(predictions,real)
        accuracy_t = np.count_nonzero(correct)/trainData.shape[1]
        accuracy_train.append(accuracy_t)

        z_h_v = np.matmul(W_h.T,validData).T + b_h
        h_v = relu(z_h_v)
        o = np.matmul(h_v,W_o) + b_o
        p_v = softmax(o)
        loss_v = averageCE(validTarget, p_v)
        loss_valid.append(loss_v)
        predictions = np.argmax(p_v,axis=1)
        real = np.argmax(validTarget,axis=1)
        correct = np.equal(predictions,real)
        accuracy_v = np.count_nonzero(correct)/validData.shape[1]
        accuracy_valid.append(accuracy_v)

        print("Epoch %d" %(epoch))
        print("   Training Loss: %.4f" %(loss_t))
        print("   Training Accuracy: %.4f" %(accuracy_t))
        print("   Validation Loss: %.4f" %(loss_v))
        print("   Validation Accuracy: %.4f" %(accuracy_v))

        v_out_weight = gamma*v_out_weight + lr*grad_o_w(trainTarget,p,h)
        v_hidden_weight = gamma*v_hidden_weight + lr*grad_h_w(trainTarget,p,trainData,z_h,W_o)
        v_out_bias = gamma*v_out_bias + lr*grad_o_b(trainTarget,p)
        v_hidden_bias = gamma*v_hidden_bias + lr*grad_h_b(trainTarget,p,z_h,W_o)

        W_o -= v_out_weight
        W_h -= v_hidden_weight
        b_o -= v_out_bias
        b_h -= v_hidden_bias
    
    plt.plot(accuracy_train, label='Training')
    plt.plot(accuracy_valid, label='Validation')
    plt.legend()
    plt.title('Accuracy vs. Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    
    plt.plot(loss_train, label='Training')
    plt.plot(loss_valid, label='Validation')
    plt.legend()
    plt.title('Loss vs. Number of Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    return W_o, W_h, b_o, b_h

if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainData = trainData.reshape((-1,trainData.shape[1]*trainData.shape[2])).T
    validData = validData.reshape((-1,validData.shape[1]*validData.shape[2])).T
    testData = testData.reshape((-1,testData.shape[1]*testData.shape[2])).T

    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

    hidden_units = 1000
    gamma = 0.99
    lr = 0.00000003
    epochs = 200

    mean = 0
    std_o = np.sqrt(2/(hidden_units+10))
    std_h = np.sqrt(2/(trainData.shape[0]+hidden_units))
    W_o = np.random.normal(mean,std_o,(hidden_units,10))
    v_o = np.full((hidden_units,10),1e-5)
    W_h = np.random.normal(mean,std_h,(trainData.shape[0],hidden_units))
    v_h = np.full((trainData.shape[0],hidden_units),1e-5)
    b_o = np.zeros((1,10))
    b_h = np.zeros((1,hidden_units))

    W_o, W_h, b_o, b_h = train(epochs, trainData, trainTarget, validData, validTarget, W_o, v_o, b_o, W_h, v_h, b_h, gamma, lr)
    
    z_h = np.matmul(W_h.T,testData).T + b_h
    h = relu(z_h)
    o = np.matmul(h,W_o) + b_o
    p = softmax(o)
    loss = averageCE(testTarget, p)
    predictions = np.argmax(p,axis=1)
    real = np.argmax(testTarget,axis=1)
    correct = np.equal(predictions,real)
    accuracy = np.count_nonzero(correct)/testData.shape[1]

    print("Test Accuracy: %.4f" %(accuracy))
    print("Test Loss: %.4f" %(loss))