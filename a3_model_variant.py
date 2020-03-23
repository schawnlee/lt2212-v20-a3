import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
from torch.autograd import Variable
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
# Whatever other imports you need

# You can implement classes and helper functions here too.


def sampler(df, batch_size=5, train=True):
    samples= []
    tensor_label_pairs=[]
    trainflag = None
    if train:
        trainflag = 0
    else:
        trainflag = 1
    for i in range(batch_size):
        authors = list(set(df["author"]))
        seed_authors = random.choices(authors, k=2)
        data_train = df[df["test_train"] == trainflag]
        seed_entries_1 = (data_train[data_train["author"] == seed_authors[0]]).sample(n=2, replace=True)
        seed_entries_2 = (data_train[data_train["author"] == seed_authors[1]]).sample(n=2, replace=True)
        if random.random() < 0.5:
            samples.append((seed_entries_1.iloc[0], seed_entries_1.iloc[1],0))
        else:
            samples.append((seed_entries_1.iloc[0], seed_entries_2.iloc[0],1))
    width = len(samples[0][0])-3
    for sample in samples:
        a = list(sample[0])[1:-2]
        b = list(sample[1])[1:-2]
        c = Variable(torch.Tensor((a+b)))
        label = torch.Tensor([sample[2]])
        tensor_label_pairs.append((c,label))
    return tensor_label_pairs
        


def pred(inputs):
    if net(inputs)>0.5:
        return 1
    else:
        return 0
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    parser.add_argument("--batch", "-B", dest="batch", type=int, default=20, help="Batch size")
    parser.add_argument("--epoch", "-E", dest="epoch", type = int, default=100, help="Batch size.")
    parser.add_argument("--hidden_size", "-H",dest="hidden", type = int, default = 0, help = "The size of hidden layer.")
    parser.add_argument("--activation_fn", "-A", dest="fn", type = str, default = None, help = "Name of the non-linear function.")
    
    args = parser.parse_args()
    
    function_map = {"sigmoid": nn.Sigmoid,"tahn": nn.Tanh, "relu": nn.ReLU}
    
    
    print("Reading {}...".format(args.featurefile))
    
    data = pd.read_csv(args.featurefile)
    width = len(data.columns)-3
    hidden_size = args.hidden
    if args.fn:
        activation_fn =  function_map[args.fn]
    else:
        activation_fn = None
        
    batch_size = args.batch
    xs= [20,30,40,50,55,60,70,80,90,100]
    recalls = []
    precisions = []

    for size in xs:
        print("Processing; size of the Hidden Layer:", size)
    
        class MyNet(nn.Module):
    
            def __init__(self, input_size, size, activation_fn):
                super(MyNet,self).__init__()
                if hidden_size == 0:
                    self.fc1 = nn.Linear(input_size, 1)
                else:
                    self.fc1 = nn.Linear(input_size, hidden_size)
                if activation_fn:
                    self.nonlinear = activation_fn()
                if hidden_size != 0:
                    self.fc2 = nn.Linear(size, 1)
                self.sigmoid = nn.Sigmoid()
        
    
            def forward(self, x):
                    x = self.fc1(x)
                    if activation_fn:
                        x = self.nonlinear(x)
                    if hidden_size != 0:
                        x = self.fc2(x)
                    x = self.sigmoid(x)
                    return x
            
            
            
        net = MyNet(width*2, hidden_size, activation_fn)
    
            
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.BCELoss()

        for e in range(args.epoch):
    
            batch_data = sampler(data, batch_size, train=True)
            loss_acc = 0
            for inputs, label in batch_data:
                optimizer.zero_grad()
                out = net(inputs)
                loss = criterion(out, label)
                loss_acc += loss
                loss.backward()
                optimizer.step()
        
        test_samples = sampler(data, 100, train=False)
        ins = [inputs for inputs, label in test_samples]
        labels = [label for inputs, label in test_samples]
    
        preds = []
        for inputs in ins:
            preds.append(pred(inputs))
        recalls.append(recall_score(labels,preds))
        precisions.append(precision_score(labels,preds))
    print(recalls) 
    print(precisions)
    plot_data = pd.DataFrame({"x":xs, "rec": recalls, "pre": precisions})
    plt.plot( 'x', 'rec', data=plot_data, marker='o', markerfacecolor='blue', markersize=12, color='skyblue',linewidth=4,label="recalls")
    plt.plot( 'x', 'pre', data=plot_data, marker='', color='olive', linewidth=2, label="precisions")
    plt.legend(('Recalls', 'Precisions'),
           loc='upper right')
    plt.savefig('trends.png')

    



    
    
    
