# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 10:20:01 2021

Aim of this program is process a raw dataset of spam messages to allow machine learning.

@author: Ross Mates
"""

import numpy as np
import pandas as pd     


file = open('spam.csv',mode='r')
rawData = file.readlines()
file.close()


rawData.pop(0) # remove the first line
messages = []
labels = [] # Makes empty lists


for line in rawData:
    comma = line.find(',')
    label = line[0:comma] # Takes data between position 0 and the comma.
    message = line[comma+1:] # Takes data after comma.
    messages.append(message)
    labels.append(label)
    
spamWords = ['free','buy','win'] # The words we are looking for when finding spam.

features = np.zeros((len(messages),len(spamWords)))

for i, message in enumerate(messages):
    # Goes through each message one by one. 'Enumerate' returns the index of each message.
    for j, word in enumerate(spamWords):
        # This goes through each of the entry in the spamWords variable.
        message = message.lower()
        wordCount = message.count(word)
        features[i,j] = wordCount

labels = np.array(labels)
features = np.array(features) # Converts to numpy array.

labels = labels.reshape((-1, 1))

dataset = np.hstack((features,labels))# Stacks them together.


# Saves to disk to be used for machine learning.

df = pd.DataFrame(dataset)
spamWords.append('label')
pd = df.to_csv(r'spam_dataset.csv', header=spamWords,index=False)