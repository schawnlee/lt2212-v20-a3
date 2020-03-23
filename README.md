# LT2212 V20 Assignment 3

PART 1
The a3_features.py file takes the path of corpus directory and outputs a .csv
file with reduced dimensions with splited train and test sets and author labels.
All of the respresentations are numeric.
A autor-index-map is printed out on the screen.
{train:0; data:1 }

Arguments:  
    "inputdir", type=str; "The root of the author directories."
    "outputfile", type=str; "The name of the output file containing the table of instances."
    "dims"; type=int; "The output feature dimensions."
    "--test", "-T", dest="testsize", type=int, default="20"; "The percentage (integer) of instances to label as test."

The challenging part is that the quality of the corpus. The corpus consitsts of
e-mail communication. In the e-mails there are also head and origincal messages
from other correspondants. I removed these to reduce the impact of irrelavant
information. This should be a meaningful meassure, easpecially if the main message
is short and the message from the others is long. However, this led to another
problem, that there are e-mails, which contain no text main message(maybe the message
body is a pictrue or a link). Some of the data entries only contain 0.00. This turned
out to have negative influence of the performance.

Part 2/Part 3

The a3_model.py file contains two variations of neural networks. The choices are
to be controlled with arguments.
model a) is a simple model only with 1 linear layer and a sigmoid function as
activation function.
model b) is a model with 1 hidden layer with flexible size and a extra layer of
non-linearal layer.
The exmaples for arguements inputs for both models are given below.

Arguments:
   "featurefile"; type=str; "The file containing the table of instances and features."
   "--batch"; "-B"; dest="batch"; type=int; default=20, help="Batch size"
   "--epoch", "-E"; dest="epoch"; type = int, default=100, help="Batch size."
   "--hidden_size"; "-H",dest="hidden"; type = int, default = 0, help = "The size of hidden layer."
   "--activation_fn"; "-A", dest="fn"; type = str, default = None, help = "Name of the non-linear function from [tahn, sigmoid, relu]"

Example:
a) python3 a3_model.py [path]
b) python3 a3_model.py [path] --batch 30 --epoch 200 --hidden_size 40 --activation_fn tahn  

Performance Report:

Single Layer NN with reduced dim_num=20 batch_size=30 epoch=100:
precision    recall  f1-score   support

SAME          0.25      0.03      0.06        29
NOT-SAME      0.39      0.86      0.54        21

accuracy                          0.38        50
macro avg     0.32      0.45      0.30        50
weighted avg  0.31      0.38      0.26        50

1 hidden Layer with size 40 NN without non-linear function with reduced dim_num=40 batch_size=30 epoch=100:
precision    recall  f1-score   support

precision    recall  f1-score   support

SAME         0.00      0.00      0.00        24
NOT-SAME     0.50      0.92      0.65        26

accuracy                         0.48        50
macro avg    0.25      0.46      0.32        50
weighted avg 0.26      0.48      0.34        50

1 hidden Layer with size 40 NN with relu as non-linear function and reduced dim_num=40 batch_size=30 epoch=100:
precision    recall  f1-score   support

SAME         0.41      0.81      0.55        21
NOT-SAME     0.56      0.17      0.26        29

accuracy                         0.44        50
macro avg    0.49      0.49      0.41        50
weighted avg 0.50      0.44      0.38        50

1 hidden Layer with size 40 NN with tahn as non-linear function and reduced dim_num=40 batch_size=30 epoch=100:
precision    recall  f1-score   support

precision    recall  f1-score   support

SAME         0.48      1.00      0.65        24
NOT-SAME     0.00      0.00      0.00        26

accuracy                         0.48        50
macro avg    0.24      0.50      0.32        50
weighted avg 0.23      0.48      0.31        50


PART PLOTTING
The a3_model_varaint file
See trands.png

Example Arguments:
python3 lt2212-v20-a3-master/data.csv --batch 20 --epoch 100 --hidden_size 40 --activation_fn relu
