import pandas as pd
import numpy as np
import math
import copy
from tensorflow.keras.datasets import mnist

np.random.seed(30)
def get_accuracy(DT, X):
    result = DT.predict(X)
    right = (result == X["label"])
    count = 0
    for i in right:
        if (i == True):
            count += 1
    return count/len(result)
def discrete_check(X_train):
    for l in X_train.columns:
        if (isinstance(X_train[l][X_train.index[0]],float)):
            minimum = min(list(X_train[l]))
            maximum = max(list(X_train[l]))
            threshold = (minimum+maximum)/2
            series = X_train[l]>threshold
            X_train.loc[:,l] = series
def bfs(root):
    current = list()
    next_nodes = list()
    whole = list()
    if (not root.is_leaf):
        current.append(root)
        whole.append(current)
    while(len(current)>0):
        for i in range(len(current)):
            for key in current[i].children.keys():
                if (not current[i].children[key].is_leaf):
                    next_nodes.append(current[i].children[key])
        current = copy.copy(next_nodes)
        whole.append(current)
        next_nodes.clear()
    return whole
def check_depth(head):
    counter = 0
    while (head.parent != None):
        head = head.parent
        counter += 1
    return counter
def TreeGenerate(X_train,feature, head,depth):
    if (check_depth(head)>depth):
        mark = poll(X_train["label"])
        head.mark_as_leaf(mark)
        return
    if (same_class(X_train)):
        head.mark_as_leaf(X_train["label"][X_train.index[0]])
        return
    if len(feature) == 0 or (len(feature) == 1 and same_class(X_train)):
        mark = poll(X_train["label"])
        head.mark_as_leaf(mark)
        return
    selected_feat = dict()
    for feat in feature:
        selected_feat[feat] = entropy_gain(X_train,feat)
    max_num = list(selected_feat.keys())[0]
    for key in selected_feat.keys():
        if selected_feat[max_num]<=selected_feat[key]:
            max_num = key
    head.set_feature(max_num)
    all_values = set()
    for values in X_train[max_num]:
        all_values.add(values)
    for single_value in all_values:
        new_head = Node()
        head.set_children(new_head,single_value)
        for k in X_train["label"]:
            head.sample.append(k)
        D_v = X_train[X_train[max_num] == single_value]
        if len(D_v) == 0:
            mark = poll(X_train["label"])
            new_head.mark_as_leaf(mark)
            return
        else:
            pop_val = max_num
            feature.remove(max_num)
            new_head.parent = head
            TreeGenerate(D_v,feature,new_head,depth)
            feature.append(pop_val)
def same_class(X_train):
    flag = True
    mark = X_train["label"][X_train.index[0]]
    for i in X_train["label"]:
        if mark != i:
            flag = False
    return flag
def same_value(X_train,feature):
    flag = True
    mark = X_train[feature][0]
    for i in X_train[feature]:
        if mark == i:
            flag = False
    return flag
def poll(series):
    stat = dict()
    for i in series:
        stat[i] = 0
    for i in series:
        stat[i] += 1
    max_num = list(stat.keys())[0]
    for key in stat.keys():
        if stat[max_num]<stat[key]:
            max_num = key
    return max_num
def entropy(label):
    label_dict = dict()
    entropy_d = 0
    sums = 0
    for i in label:
        label_dict[i] = 0
    for i in label:
        label_dict[i] += 1
    for key in label_dict.keys():
        sums += label_dict[key]
    for key in label_dict.keys():
        label_dict[key] = label_dict[key]/sums
    for key in label_dict.keys():
        entropy_d -= label_dict[key]*math.log2(label_dict[key])
    return entropy_d
def entropy_gain(X_train,feature):
    entropy_d = 0
    sums = 0
    label = X_train["label"]
    entropy_d = entropy(label)
    Gain = dict()
    temp = dict()
    sum_feature = 0
    gain_on_each = 0
    for l in X_train[feature]:
        temp[l] = 0
    for l in X_train[feature]:
        temp[l] += 1
    for key in temp.keys():
        sum_feature += temp[key]
    for key in temp.keys():
        gain_on_each -= (temp[key]/sum_feature)*entropy(label[X_train[feature] == key])
    return entropy_d+gain_on_each       
    
class Node:
    def __init__(self):
        self.feature = ""
        self.children = dict()
        self.is_leaf = False
        self.root = None
        self.parent = None
        self.sample = list()
    def mark(self,mark):
        assert(self.is_leaf == False)
        self.feature = mark
    def mark_as_leaf(self,value):
        self.children["value"] = value
        self.is_leaf = True
    def set_feature(self,feature):
        self.feature = feature
    def set_children(self,child,feature):
        self.children[feature] = child

class DecisionTree:
    def __init__(self,max_depth = 50, pruning = False):
        self.max_depth = max_depth
        self.pruning = pruning
    def fit(self,X_train,feature):  
        discrete_check(X_train)
        head = Node()
        if (self.pruning):
            msk = np.random.rand(len(X_train)) < 0.8
            train = X_train[msk]
            test = X_train[~msk]
            self.tree = TreeGenerate(train,feature,head,self.max_depth)
        else:
            self.tree = TreeGenerate(X_train,feature,head,self.max_depth)
        self.root = head 
        if (self.pruning):
            pruning_nodes = bfs(self.root)
            pruning_nodes.reverse()
            for l in range(len(pruning_nodes)): 
                for k in range(len(pruning_nodes[l])):
                    accuracy_before = get_accuracy(self, test)
                    mark = poll(pruning_nodes[l][k].sample)
                    pruning_nodes[l][k].mark_as_leaf(mark)
                    if (get_accuracy(self,test)<accuracy_before):
                        del pruning_nodes[l][k].children["value"]
                        pruning_nodes[l][k].is_leaf = False
                    else:
                        keys = list(pruning_nodes[l][k].children.keys())
                        for j in keys:
                            if (j != "value"):
                                del pruning_nodes[l][k].children[j]
                                pruning_nodes[l][k].feature = ""
    def predict(self,data_frame):
        discrete_check(data_frame)
        result = list()
        features = list(data_frame.columns)
        features.remove("label")
        root = self.root
        for index,row in data_frame.iterrows():
            while (not (root.is_leaf)):
                next_index = features.index(root.feature)
                if (not row[next_index] in list(root.children.keys())):    
                    root = root.children[list(root.children.keys())[0]]
                else:
                     root = root.children[row[next_index]]
            result.append(root.children["value"])
            root = self.root
        return result
                
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = pd.DataFrame(X_train.reshape(60000,784))
y_train = pd.DataFrame(y_train.reshape(60000,1))
y_train.columns = ["label"]
DT = DecisionTree(pruning = True)
X_test = pd.DataFrame(X_test.reshape(10000,784))
y_test = pd.DataFrame(y_test.reshape(10000,1))
y_test.columns = ["label"]
train = pd.concat([X_train,y_train],axis = 1).head(1000)
test = pd.concat([X_test,y_test],axis = 1).head(50)
DT = DecisionTree(pruning = True)
features = list(train.columns)
features.remove("label")
DT.fit(train,features)
result = DT.predict(test)
accuracy = get_accuracy(DT, test)
print(accuracy)