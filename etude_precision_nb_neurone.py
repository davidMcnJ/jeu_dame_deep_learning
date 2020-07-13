# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:47:47 2020

@author: David Micouin--Jorda

"""

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow
import numpy as np
import pickle
import time

import sys
import fonction_apprentissage

import matplotlib.pyplot as plt
import math


def mean_square(X, Y):
    

  mean_square = 0
  for i in range(len(Y)):
    mean_square += (Y[i][0] - X[i][0])**2
  mean_square /= len(Y)
  return mean_square


#Training examples
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


tmps1=time.clock()

list_mean_square = []
list_mean_square_val = []
list_mean_square_test = []
listNbNeuronesParCouche = [1, 2, 5, 10, 20, 30, 50]
for nbNeuronesParCouche in listNbNeuronesParCouche:
  #fit the model

  tensorflow.reset_default_graph()
    
    
  input_layer = input_data(shape=[None, 72]) #input layer of size 2
  hidden_layer = fully_connected(input_layer , nbNeuronesParCouche, activation='tanh') #hidden layer of size 2
  output_layer = fully_connected(hidden_layer, 1, activation='linear') #output layer of size 1
    
  #use Stohastic Gradient Descent and Binary Crossentropy as loss function
  maRegression = regression(output_layer , optimizer='sgd', loss='mean_square', learning_rate=0.1)
  model = DNN(maRegression, checkpoint_path='/tmp/tflearn_logs/',max_checkpoints=1, tensorboard_verbose=0)
  
  
  model.fit(X, Y, validation_set=(Xval, Yval),  n_epoch=50, show_metric=True);

  list_mean_square.append(mean_square(Y, model.predict(X)))
  list_mean_square_val.append(mean_square(Yval, model.predict(Xval)))
  list_mean_square_test.append(mean_square(Ytest, model.predict(Xtest)))
  model.save("./modele/model_etude_precision_nb_neurone_" + str(nbNeuronesParCouche))


tmps2=time.clock()
print("Temps d'execution = ", (tmps2-tmps1) ,"\n")

 
#predict all examples
print ('Expected:  ', [i[0]  for i in Y])
print ('Predicted: ', [i[0]  for i in model.predict(X)])

print(list_mean_square)

plt.figure(1)
plt.plot(listNbNeuronesParCouche, list_mean_square)
plt.plot(listNbNeuronesParCouche, list_mean_square_val)
plt.plot(listNbNeuronesParCouche, list_mean_square_test)
#model.get_weights(hidden_layer2.W)
#model.get_weights(output_layer.W)
 
#model.save("tflearn-xor")