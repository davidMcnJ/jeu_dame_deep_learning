# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:57:19 2020

@author: david
"""


import numpy as np
import math
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import fonction_apprentissage
import fonction_basique_jeu

def choisirMeilleurCoupModeleTensorflow(modele, listePlateau, taillePlateau):
    xVecteur = fonction_apprentissage.convertirTousPlateauxVecteur(listePlateau, taillePlateau)

    sortieModele = modele.predict(xVecteur)
    #print(sortieModele)
    indiceMeilleurCoup = np.argmax(sortieModele)
    return indiceMeilleurCoup

#anciennemthode ; peu concluante
def choisirMeilleurCoupModeleScikitLearn(modele, acpCharge1, nbComposantesAcp1, acpCharge2, nbComposantesAcp2, deg, listePlateau, taillePlateau):
    xVecteur = fonction_apprentissage.convertirTousPlateauxVecteur(listePlateau, taillePlateau)
    
    
#    xVecteurTransforme1 = acpCharge1.transform(xVecteur)
#    xVecteurTransforme1 = xVecteurTransforme1[:, 0:nbComposantesAcp1] 
    
    poly = PolynomialFeatures(deg)
    xPoly = poly.fit_transform(xVecteur)
    
#    xVecteurTransforme2 = acpCharge2.transform(xPoly)
#    xVecteurTransforme2 = xVecteurTransforme2[:, 0:nbComposantesAcp2] 
    
    sortieModele = modele.predict(xPoly)
    indiceMeilleurCoup = np.argmax(sortieModele)
    return indiceMeilleurCoup

def choisirPlateauIaArbre(taillePlateau, listePlateau, rec, recInit, couleurInit, couleurActu):
    valeurIndicePlateauGagnant = []
    if(rec == recInit):
        for p in range(len(listePlateau)):
            osef, listePlateauSuivant =fonction_basique_jeu.definirDeplacementPossible(taillePlateau, listePlateau[p], -couleurActu)
            valeurIndicePlateauGagnant.append(choisirPlateauIaArbre(taillePlateau, listePlateauSuivant, rec-1, rec, couleurInit, -couleurActu))
        if(couleurInit > 0):
            valRetour = np.amax(valeurIndicePlateauGagnant)
            listeMax = np.argwhere(valeurIndicePlateauGagnant == np.amax(valeurIndicePlateauGagnant)).flatten().tolist()
#            numPlateau = np.argmax(valeurIndicePlateauGagnant)
            numPlateau = listeMax[fonction_basique_jeu.choixRandom(listeMax)]
        else:
            valRetour = np.amin(valeurIndicePlateauGagnant)
            listeMin = np.argwhere(valeurIndicePlateauGagnant == np.amin(valeurIndicePlateauGagnant)).flatten().tolist()
            numPlateau = listeMin[fonction_basique_jeu.choixRandom(listeMin)]
        return numPlateau, valRetour
    elif(rec != 0):
        if(len(listePlateau) == 0):
            return -couleurActu*taillePlateau
        for p in range(len(listePlateau)):
            osef, listePlateauSuivant = fonction_basique_jeu.definirDeplacementPossible(taillePlateau, listePlateau[p], -couleurActu)
            valeur = choisirPlateauIaArbre(taillePlateau, listePlateauSuivant, rec-1, rec, couleurInit, -couleurActu)
            valeurIndicePlateauGagnant.append(valeur)
        if(couleurActu > 0):
            return max(valeurIndicePlateauGagnant)
        else:
            return min(valeurIndicePlateauGagnant)
    elif(rec == 0):
        valeurPlateau = []
        if(len(listePlateau) == 0):
            return -couleurActu*taillePlateau
        for p in listePlateau:
            valeurPlateau.append(sum(sum(p)))
        if(couleurActu > 0):
            return max(valeurPlateau)
        else:
            return min(valeurPlateau)
            