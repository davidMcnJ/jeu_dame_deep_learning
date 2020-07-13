# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:54:19 2020

@author: David Micouin--Jorda

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

def convertirPlateauVecteur(plateau, taillePlateau):
    """converti le plateau en une liste de taille i^2
    où les i^2/2 premiers éléments diront où l'équipe 1 a des pions sur l'ensemble des cases 
    (en partant de en haut à gauche)
    et les i^2/2 suivant pour les pions -1
    """
    plateauVecteur = [0 for i in range((taillePlateau**2)*2)]
    c = 0
    for i in range(taillePlateau):
        for j in range(taillePlateau):
            if( (i+j)%2 == 0):
                if(plateau[i, j] == 1):
                    plateauVecteur[c] = 1
                elif(plateau[i, j] == 2):
                    plateauVecteur[int(taillePlateau**2/2) + c] = 1
                elif(plateau[i, j] == -1):
                    plateauVecteur[int(taillePlateau**2) + c] = 1
                elif(plateau[i, j] == -2):
                    plateauVecteur[int(3*taillePlateau**2/2) + c] = 1
                c += 1
                    
    return plateauVecteur

def convertirTousPlateauxVecteur(listePlateau, taillePlateau):
    listeVecteurPlateaux = []
    for p in listePlateau:
        listeVecteurPlateaux.append(convertirPlateauVecteur(p, taillePlateau))
    return listeVecteurPlateaux

def apprendreModele(listePlateauEquipe1, y, deg, taillePlateau):
#    input("continuer ?")
    print("Conversion des plateaux en vecteurs...")
    xVecteur = convertirTousPlateauxVecteur(listePlateauEquipe1, taillePlateau)
    print("ok")
#    input("continuer ?")
    
#    print("Récupération des composantes principales...")    
#    sc = StandardScaler()
#    acp1 = PCA(svd_solver='full')
#    zVecteur = sc.fit_transform(xVecteur)
#    xVecteurProj = acp1.fit_transform(zVecteur)
#    n = len(zVecteur)
#    p = len(zVecteur[0])
#    eigval = (n-1)/n*acp1.explained_variance_
#    plt.plot(np.cumsum(eigval)/sum(eigval))    
#    sommeCum = 0
#    for i1 in range(len(eigval)):
#        sommeCum = sommeCum + eigval[i1]
#        if (sommeCum/sum(eigval) > 0.95):
#            break
#    print(i1)
#    xVecteurProjPrincipal = xVecteurProj[:, 0:i1]
    
    print("Transformation pour polynome...")
    poly = PolynomialFeatures(deg)
    xPoly = poly.fit_transform(xVecteur)
    print("ok")
#    input("continuer ?")
    
    
    
#    print("Récupération des composantes principales...")    
#    sc = StandardScaler()
#    acp2 = PCA(svd_solver='full')
#    zVecteurPoly = sc.fit_transform(xPoly)
#    xVecteurProj = acp2.fit_transform(zVecteurPoly)
#    n = len(zVecteurPoly)
#    p = len(zVecteurPoly[0])
#    eigval = (n-1)/n*acp2.explained_variance_
#    plt.plot(np.cumsum(eigval)/sum(eigval))    
#    sommeCum = 0
#    for i2 in range(len(eigval)):
#        sommeCum = sommeCum + eigval[i2]
#        if (sommeCum/sum(eigval) > 0.99):
#            break
#    print(i2)
#    xVecteurProjPrincipalPoly = xVecteurProj[:, 0:i2]
    
    print("ok")
#    input("continuer ?")
    
    
    print("Régression...")
    modele = LinearRegression()
    modele.fit(xPoly,y)
    print("ok")
#    input("continuer ?")
#    return modele, acp1, i1, acp2, i2
    return modele, 0, 0, 0, 0