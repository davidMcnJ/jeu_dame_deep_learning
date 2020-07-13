# -*- coding: utf-8 -*-
"""

David Micouin--Jorda

Jeu de dame par Machine Learning
"""

import tensorflow as tf
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import numpy as np
import math
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


import fonction_basique_jeu
import IA_decision
import fonction_apprentissage

"""
Idée : 
    ACP : pas forcément très utile...
    Au lieu de passer à des vecteurs 0/1 pour chaque pièces (vecteurs de taille 4*nb de cases actives)
    Utiliser des valeurs -1/1 pour les pions et 2/-2 pour les dames (ou -3/3,
        etc quite à faire une apprentissage pour trouver la meilleure valeur) 
        dans le but de pouvoir donner plus d'échantillons pour moins de paramètres
    sélectionner de manière plus pertinentes les plateaux et leurs valeurs pour l'apprentissage
    Faire du k-fold
    Réaliser plusieurs régressions pour avoir plusieurs modèles, algorithme de décision avec un système de vote
    valeur des plateaux vers 1 ou -1 en suivant une parabole (bien valoriser la victoire/défaite)
    -> Peu concluant
    
    -> Tester avec plus de données avec réseaux de neurones profonds
    

"""          
      
        


deg = 1
rec = 3
taillePlateau = 6
plateau = fonction_basique_jeu.initierPlateau(taillePlateau)
nbPerduUn = 0
nbGagneUn = 0
nbSimul = 100000
nbCoupJoue = 0
nbCoupJouePartie = 0
modeJeu = 3 #1 = manuel, 2 = random,3=intelligent brut
deepIA = False
nbNeuronesParCouche = 50

#ancien modele scikit learn
#if (deepIA == True):
#    modeleCharge = pickle.load(open("IA8Cases_2", 'rb'))
if (deepIA == True):
    tf.reset_default_graph()
    input_layer = input_data(shape=[None, 72]) #input layer of size 2
    hidden_layer1 = fully_connected(input_layer , nbNeuronesParCouche, activation='tanh') #hidden layer of size 2
    #hidden_layer2 = fully_connected(hidden_layer1 , nbNeuronesParCouche, activation='tanh') #hidden layer of size 2
    output_layer = fully_connected(hidden_layer1, 1, activation='linear') #output layer of size 1
    
    #use Stohastic Gradient Descent and Binary Crossentropy as loss function
    maRegression = regression(output_layer , optimizer='sgd', loss='mean_square', learning_rate=0.01)
    modeleCharge = DNN(maRegression, checkpoint_path='/tmp/tflearn_logs/',max_checkpoints=1, tensorboard_verbose=0)
    modeleCharge.load('./50_neurones_2_hidden_layers_1000epochs_5000parties_rec3/model')
    
listePlateauEquipe1Partie = []
listePlateauEquipe1 = []
valeurPlateauPartie = []
valeurPlateau = []

for i in range(nbSimul):
    plateau = fonction_basique_jeu.initierPlateau(taillePlateau)
    perdant = 0  
    while(1):
        for couleur in [1, -1]:
            nbCoupJoue += 1
            nbCoupJouePartie += 1
            listeDeplacement, listePlateau = fonction_basique_jeu.definirDeplacementPossible(taillePlateau, plateau, couleur)
       
                
            if(len(listePlateau) == 0):
                print(i, end="")
                if(couleur == 1):
                    print(" MAAAAAAL          ", end="")
                else:
                    print(" BIEEEEEN", end="")
                print(" : l'équipe ", couleur, " a perdu")
                perdant = couleur
                break
            if couleur == 1:
                listePlateauEquipe1Partie.append(plateau)
            
            if(couleur == -1):
                if(modeJeu == 1):
                    for p in range(len(listePlateau)):
                        print(p, "\n", listePlateau[p])
#                    print(listeDeplacement, '\n', listePlateau)
                    print('\n', plateau)
                    choix = int(input(("choisissez votre coup entre 0 et " + str(len(listePlateau)-1)) ))
                    plateau = listePlateau[choix]
                elif(modeJeu == 2):
                    plateau = listePlateau[fonction_basique_jeu.choixRandom(listePlateau)]
                elif(modeJeu == 3):
                    choix, OSEF = IA_decision.choisirPlateauIaArbre(taillePlateau, listePlateau, rec, rec, couleur, couleur)
                    plateau = listePlateau[choix]

            else: 
                if(deepIA == True):#l'équipe 1 est joue en mode deep-IA
                    indice = IA_decision.choisirMeilleurCoupModeleTensorflow(modeleCharge, listePlateau, taillePlateau)
                    plateau = listePlateau[indice]
                    choix, valPrevision = IA_decision.choisirPlateauIaArbre(taillePlateau, listePlateau, rec, rec, couleur, couleur)
                    #print("prev vraie", valPrevision, "ind", choix)
                elif(modeJeu == 1):
                    for p in range(len(listePlateau)):
                        print(p, "\n", listePlateau[p])
#                    print(listeDeplacement, '\n', listePlateau)
                    print('\n', plateau)
                    choix = int(input(("choisissez votre coup entre 0 et " + str(len(listePlateau)-1)) ))
                    plateau = listePlateau[choix]
                elif(modeJeu == 2):
                    plateau = listePlateau[fonction_basique_jeu.choixRandom(listePlateau)]
                elif(modeJeu == 3):
                    choix, valPrevision = IA_decision.choisirPlateauIaArbre(taillePlateau, listePlateau, rec, rec, couleur, couleur)
                    plateau = listePlateau[choix]
                    valeurPlateauPartie.append([valPrevision])
        if(nbCoupJouePartie > 70):
            perdant = 2
            print("bloquage")
        if(perdant != 0):
            nbCoup = len(listePlateauEquipe1Partie)
            if(perdant == 1):
                nbPerduUn += 1 
#                valeurPlateauPartie = [i/nbCoup for i in range(nbCoup)]
            elif(perdant == -1):
#                valeurPlateauPartie = [-i/nbCoup for i in range(nbCoup)]
                nbGagneUn += 1 
    
            listePlateauEquipe1 = listePlateauEquipe1 + listePlateauEquipe1Partie
            #valeurPlateauPartie = [sum(sum(listePlateauEquipe1Partie[min(i+2, nbCoup-1)])) for i in range(nbCoup)]
            valeurPlateau = valeurPlateau + valeurPlateauPartie
            
            listePlateauEquipe1Partie = []
            valeurPlateauPartie = []
            nbCoupJouePartie = 0
            break
print(valeurPlateau)

print("equipe 1 : ", (nbPerduUn/nbSimul*100), "% de défaites")
print("equipe 1 : ", (nbGagneUn/nbSimul*100), "% de victoires")
print((nbCoupJoue/nbSimul), "coups moyens total joués par partie")

print("\n")

print("( ", len(listePlateauEquipe1), " plateaux pour apprendre )")


#Fichier = open('plateau.txt', 'wb')
#pickle.dump(listePlateauEquipe1, Fichier)
np.savetxt('plateaasuppr.txt', fonction_apprentissage.convertirTousPlateauxVecteur(listePlateauEquipe1, taillePlateau))
#Fichier.close()
#Fichier = open('valPlateau.txt', 'wb')
np.savetxt('valPlateaasupru.txt', valeurPlateau)
#Fichier.close()

#ancienne methode d'apprentissage avec scikit-learn, peu concluant
#print("apprentissage...")
#modele, acp1, i1, acp2, i2 = fonction_apprentissage.apprendreModele(listePlateauEquipe1, valeurPlateau, deg, taillePlateau)

print("Terminé")


