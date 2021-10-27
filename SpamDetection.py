# -*- coding: utf-8 -*-
"""
Lab 4, use the dataset 'spam_dataset.csv'.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

filename = 'spam_dataset.csv'
dataset = pd.read_csv(filename)
dataset = np.array(dataset) # Dataset goes from DataFrame to object.

features = dataset[:,0:3] # All numbers
labels = dataset[:,-1]

# Encode data. This will change the labels 'Ham' and 'Spam' to 1s and 0s.

myEncoder = LabelEncoder()
labels = myEncoder.fit_transform(labels) # Normalises labels, allows labels to use a common scale to facilitate ML.

# split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)
model = SVC(kernel='rbf',probability=True) # Radial Basis Function
# Probability = True makes the model calculate the individual probability of each sample being 'Ham' or 'Spam'.
model.fit(train_features, train_labels)

# Prints the results.

trainAcc = float(model.score(train_features,train_labels))*100 # Gets the accuracy of the training in a percent.
print('Accuracy of train is: ' + str("%.2f" % trainAcc) + '%')
# Prints the accuracy of the training, formatted in percent and to two decimal places.
print(str(model.fit_status_) + ' errors')

probs = model.predict_proba(test_features[0:10,:]) # Gauges how confident the ML is at catagorising.
print('previous 10 attempts:')
print(probs)

# Predictions

testLabelPredictions = model.predict(test_features) # The predict method returns the predicted class labels.
accScore = float(accuracy_score(test_labels,testLabelPredictions))*100
print('Accuracy of predict is: ' + str("%.2f" % accScore) + '%')

precScore = float(precision_score(test_labels,testLabelPredictions))*100
print('Prediction score is: ' + str("%.2f" % precScore) + '%')

recallScore = float(recall_score(test_labels,testLabelPredictions))*100
print('Recall is: ' + str("%.2f" % recallScore) + '%')

# Save trained model

file = open('TrainedSpamModel.pkl',mode='wb')
pickle.dump(model,file)
file.close()

