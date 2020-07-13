# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 02:36:10 2020

@author: David Micouin--Jorda

"""

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow
import numpy as np

import matplotlib.pyplot as plt

import math


def mean_square(X, Y):
    

  mean_square = 0
  for i in range(len(Y)):
    mean_square += (Y[i][0] - X[i][0])**2
  mean_square /= len(Y)
  return mean_square



nbNeuronesParCouche = 30
tpsApprentissage = []

Xdata = np.loadtxt('plateau.txt')
Ydata = np.loadtxt("valPlateau.txt")
Ydata = np.array(Ydata)
Ydata = Ydata.reshape(len(Ydata), 1)


nbDonnee = Xdata.shape[0]
endDataTrain = math.floor(nbDonnee*0.8)
endDataVal = math.floor(nbDonnee*0.9)
X = Xdata[0:endDataTrain, :]
Y = Ydata[0:endDataTrain, :]
Xval = Xdata[endDataTrain:endDataVal, :]
Yval = Ydata[endDataTrain:endDataVal, :]
Xtest = Xdata[endDataVal:nbDonnee-1, :]
Ytest = Ydata[endDataVal:nbDonnee-1, :]

print(X.shape)

tensorflow.reset_default_graph()
input_layer = input_data(shape=[None, 72]) #input layer of size 2
hidden_layer = fully_connected(input_layer , nbNeuronesParCouche, activation='tanh') #hidden layer of size 2
output_layer = fully_connected(hidden_layer, 1, activation='linear') #output layer of size 1

#use Stohastic Gradient Descent and Binary Crossentropy as loss function
maRegression = regression(output_layer , optimizer='sgd', loss='mean_square', learning_rate=0.01)
model = DNN(maRegression, checkpoint_path='/tmp/tflearn_logs/',max_checkpoints=1, tensorboard_verbose=0)
model.load("./modele/model_entrainement_reseau_neurone_learning_rate_0_11")


list_mean_square = []
list_mean_square_val = []
list_mean_square_test = []
list_mean_square.append(mean_square(Y, model.predict(X)))
list_mean_square_val.append(mean_square(Yval, model.predict(Xval)))
list_mean_square_test.append(mean_square(Ytest, model.predict(Xtest)))

plt.figure(1)
plt.plot(list_mean_square)
plt.plot(list_mean_square_val)
plt.plot(list_mean_square_test)




