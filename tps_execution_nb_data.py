# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:01:16 2020

@author: david
"""

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow
import numpy as np
import pickle
import time

import sys
sys.path.append("drive/My drive/jeu_dame/")
import fonction_apprentissage


#Training examples
nbNeurones = 100
tpsApprentissage = []

Xdata = np.loadtxt('plateau.txt')
Ydata = np.loadtxt("valPlateau.txt")
Ydata = np.array(Ydata)
Ydata = Ydata.reshape(len(Ydata), 1)

for i in range(50, len(Ydata), 100):

  X = Xdata[0:i, :]
  Y = Ydata[0:i, :]
  print(X.shape)
  
  print("Nombre de données : ", i)

  tensorflow.reset_default_graph()
  tmps1=time.clock()
  input_layer = input_data(shape=[None, 72]) #input layer of size 2
  hidden_layer1 = fully_connected(input_layer , 100, activation='tanh') #hidden layer of size 2
  hidden_layer2 = fully_connected(hidden_layer1 , 100, activation='tanh') #hidden layer of size 2
  output_layer = fully_connected(hidden_layer2, 1, activation='linear') #output layer of size 1
  
  #use Stohastic Gradient Descent and Binary Crossentropy as loss function
  maRegression = regression(output_layer , optimizer='sgd', loss='mean_square', learning_rate=0.01)
  model = DNN(maRegression)
  
  #fit the model
  model.fit(X, Y, n_epoch=5, show_metric=True);
  tmps2=time.clock()
  print("Temps d'execution = ", (tmps2-tmps1) ,"\n")
  tpsApprentissage.append((tmps2-tmps1))

print(tpsApprentissage)

#predict all examples
#print ('Expected:  ', [i[0]  for i in Y])
#print ('Predicted: ', [i[0]  for i in model.predict(X)])
 
#model.get_weights(hidden_layer2.W)
#model.get_weights(output_layer.W)
 
#model.save("tflearn-xor")