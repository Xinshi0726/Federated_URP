import pandas as pd 
import numpy as np
import math
import copy
import random
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 
def row_check(row):
    for i in range(len(row)):
        if (row[i]!=0):
            row[i] = 0
    return row 
def loss_check(network,answer,counter):
    flag = True
    sums = 0
    for i in range(len(network.network[-1])):
        if (network.network[-1][i].correct):
            sub = 1
        else:
            sub = 0
        sums += (sub-network.network[-1][i].value)**2
    if (sums>0.01):
        flag = False
    if (sums<0.5):
        network.learning_rate = 0.3
    network.loss = sums
    if (counter%50 == 0):
        print("TOTError:",sums)
    return flag
def Back_propagate(network, X, y, counter, multi):
    for index,row in X.iterrows():
        row = scale(row,0,1)
        counter += 1
        if (counter%50 == 0):
            print("epoch:{}".format(counter+multi*50))
            print("index：",index)
        prediction = network.predict_single(row,y[index])
        if (counter%50 == 0):
            print("pred: ",prediction, "actual: ",y[index])
        if (loss_check(network,y[index],counter)):
           break 
        for l in range(len(network.network)-2,-1,-1):
            the_node = network.network[l][0]
            if (list(network.network[l][0].weight_connect.keys())[0].character =="output"):
                for node in list(the_node.weight_connect.keys()): 
                    if (node.correct):
                         y_k = 1
                    else:
                        y_k = 0
                    g_j = node.value*(1-node.value)*(y_k - node.value)
                    delta_bias = -1*network.learning_rate*g_j
                    node.bias += delta_bias
                    node.loss = g_j
                    for j in range(len(network.network[l])):    
                        delta_w_hj = network.learning_rate*network.network[l][j].value*g_j
                        network.network[l][j].weight_connect[node] += delta_w_hj
            else:
                for node in list(the_node.weight_connect.keys()): 
                    sum_weight_dot_g_j = 0
                    for next_node in node.weight_connect.keys():
                        sum_weight_dot_g_j += (node.weight_connect[next_node]*next_node.loss)
                    error_h = node.value*(1-node.value)*(sum_weight_dot_g_j)
                    delta_bias = -1*network.learning_rate*error_h
                    node.loss = error_h
                    node.bias += delta_bias
                    for j in range(len(network.network[l])):    
                        delta_vih = network.learning_rate*error_h*network.network[l][j].value
                        network.network[l][j].weight_connect[node] += delta_vih             
def sigmoid(weight,value,bias):
    dot = np.dot(weight,value.transpose())
    dot -= bias
    return 1/(1+math.exp(-1*dot))

class Neural:
    def __init__(self, character = "", value = 0):
        self.value = 0
        self.character = character
        self.threshold = 0.5
        self.bias = random.uniform(0, 1)
        self.answer = ""
        self.weight_connect = dict()
        if (self.character == "hidden"):
            self.activated = False
        if (self.character == "output"):
            self.correct = False
        self.loss = -1
class NeuralNetwork:
    def __init__(self, hidden_layer = list([16,16]),learning_rate = 0.8):
        self.hidden_layer = hidden_layer
        self.learning_rate = learning_rate
    def fit(self,X_train,y):
        self.network = self.build(784, 10, y)
        self.loss = 100
        counter = 0
        multi = 0
        while (self.loss>0.01):
            Back_propagate(self,X_train,y, counter,multi)
            multi += 1
    def build(self, num_input, num_output, label):
        neuro_list = [[]]
        single_layer = list()
        network = list()
        for i in range(num_input):
            neuron = Neural(character = "input")
            network.append(neuron)
        neuro_list.append(network)
        single_layer.clear()
        for l in range(len(self.hidden_layer)):
            for j in range(self.hidden_layer[l]):
                neuron = Neural(character = "hidden")
                single_layer.append(neuron)
            temp = copy.deepcopy(single_layer)
            neuro_list.append(temp)
            single_layer.clear()
        catagorical = set()
        for i in label:
            catagorical.add(i)
        for cat in catagorical:
            neuron = Neural(character = "output")
            neuron.answer = cat
            single_layer.append(neuron)
        neuro_list.append(single_layer)
        neuro_list.pop(0)
        for ls in range(len(neuro_list)-1):
            for obj in range(len(neuro_list[ls])):
                for next_obj in range(len(neuro_list[ls+1])):
                    neuro_list[ls][obj].weight_connect[neuro_list[ls+1][next_obj]] = random.uniform(0,0.5)
        return neuro_list
    def predict_single(self,X, y = ""):
        predict = list()
        #number of hidden layers
        for l in range(len(X)):
            self.network[0][l].value = X[l]
        for i in range(len(self.network)-1):
            the_node = self.network[i][0]
            for node in the_node.weight_connect:
                weight = list()
                value = list()
                for k in range(len(self.network[i])):                    
                        weight.append(self.network[i][k].weight_connect[node])
                        value.append(self.network[i][k].value)
                sig = sigmoid(np.array(weight),np.array(value),node.bias)
                if (sig>node.threshold):
                    node.activated = True
                else:
                    node.activated = False
                node.value = sig
        for i in range(len(self.network[-1])):
             predict.append((self.network[-1][i].answer,self.network[-1][i].value))
             if (y == self.network[-1][i].answer):
                 self.network[-1][i].correct = True
             else:
                 self.network[-1][i].correct = False
        return predict   
network = NeuralNetwork()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,784)
X_train = X_train[0:100]
X_test = X_test.reshape(10000,784)
X_test = X_test[0:5]
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
network.fit(X_train,y_train)
for index,row in X_test.iterrows():  
     print(network.predict_single(row), "actual: ",y_test[index])