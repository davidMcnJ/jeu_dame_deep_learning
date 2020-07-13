# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:52:50 2020

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


def initierPlateau(taillePlateau):
    plateau =  np.zeros((taillePlateau, taillePlateau), dtype='i')
    for j in range(taillePlateau):
        for i in range(2):
            if( (i+j)%2 == 0):
                plateau[i, j] = 1
        for i in range(taillePlateau-2, taillePlateau):
            if( (i+j)%2 == 0):
                plateau[i, j] = -1
    return plateau

                
def definirDeplacementPossible(taillePlateau, plateau, couleur):
    listeDeplacement = []
    listePlateau = []
    for i in range(taillePlateau):
        for j in range(taillePlateau):
            piece = plateau[i, j]
            if(piece == 0):
                continue
            couleurPiece = int(piece/abs(piece))
            if(couleurPiece == couleur):
                listeDeplacementPiece, listePlateauPiece = definirDeplacementPossiblePiece(taillePlateau, plateau, i, j)
                listeDeplacement = listeDeplacement + listeDeplacementPiece
                listePlateau = listePlateau + listePlateauPiece
    
    listeDeplacement, listePlateau = obligerManger(plateau, listeDeplacement, listePlateau, taillePlateau, couleur)
    return listeDeplacement, listePlateau
            
def peutSauter(taillePlateau, plateau, i, j, couleurPiece, signei, signej, piece):
    if(plateau[i + signei, j + signej] * piece < 0):#<0 => signe différent => pièce ennemie
        if(existeCase(taillePlateau, i + 2*signei, j + 2*signej) and plateau[i + 2*signei, j + 2*signej] == 0):
            return True
    return False

def mangerPiece(taillePlateau, plateau, i, j, piece, couleurPiece, iInit, jInit, plateauInit):
    manger = False
    listeDeplacement = []
    listePlateau = []
    for di in [-1, 1]:
        for dj in [-1, 1]:
            if(existeCase(taillePlateau, i + di, j + dj) and peutSauter(taillePlateau, plateau, i, j, couleurPiece, di, dj, piece)):
                        manger=True
                        plateauCopy = np.copy(plateau)
                        plateauCopy[i, j] = 0
                        plateauCopy[i + di, j + dj] = 0
                        plateauCopy[ i + 2*di, j + 2*dj] = piece
                        deplacementTemp, plateauTemp = mangerPiece(taillePlateau, plateauCopy, i + 2*di, j + 2*dj, piece, couleurPiece, iInit, jInit, plateauInit)
                        
                        
                        if(listePlateau == []):
                            listeDeplacement = deplacementTemp
                            listePlateau = plateauTemp
                        else:
                            listePlateau = listePlateau + plateauTemp
                            listeDeplacement = listeDeplacement + deplacementTemp
    if(manger==False): 
        if((plateau==plateauInit).all() == False):#si on a pas pu manger maintenant mais qu'on a le meme plateau que l'initial
            return [iInit, jInit, i, j], [plateau]
        else:
            return [], []
    else:
        return listeDeplacement, listePlateau
            
            
def definirDeplacementPossiblePiece(taillePlateau, plateau, i, j):
    piece = plateau[i, j]
    couleurPiece = int(piece/abs(piece))
    listeDeplacement = []
    listePlateau = []
    #manger
    listeDeplacement, listePlateau = mangerPiece(taillePlateau, plateau, i, j, piece, couleurPiece, i, j, plateau)
    
    #déplacement simple pion
    if(existeCase(taillePlateau, i + couleurPiece, j + couleurPiece)):
        if(plateau[i + couleurPiece, j + couleurPiece] == 0):
            listeDeplacement.append([i, j, i + couleurPiece, j + couleurPiece])
            plateauCopy = np.copy(plateau)
            plateauCopy[i, j] = 0
            plateauCopy[i + couleurPiece, j + couleurPiece] = piece
            listePlateau.append(plateauCopy)
    if(existeCase(taillePlateau, i + couleurPiece, j - couleurPiece)):
        if(plateau[i + couleurPiece, j - couleurPiece]  == 0):
            listeDeplacement.append([i, j, i + couleurPiece, j - couleurPiece])
            plateauCopy = np.copy(plateau)
            plateauCopy[i, j] = 0
            plateauCopy[i + couleurPiece, j - couleurPiece] = piece
            listePlateau.append(plateauCopy)
            
    #si on a une dame
    if(existeCase(taillePlateau, i - couleurPiece, j - couleurPiece) and abs(piece) == 2):
        if(plateau[i - couleurPiece, j - couleurPiece] == 0):
            listeDeplacement.append([i, j, i - couleurPiece, j - couleurPiece])
            plateauCopy = np.copy(plateau)
            plateauCopy[i, j] = 0
            plateauCopy[i - couleurPiece, j - couleurPiece] = piece
            listePlateau.append(plateauCopy)
    if(existeCase(taillePlateau, i - couleurPiece, j + couleurPiece) and abs(piece) == 2):
        if(plateau[i - couleurPiece, j + couleurPiece] == 0):
            listeDeplacement.append([i, j, i - couleurPiece, j + couleurPiece])
            plateauCopy = np.copy(plateau)
            plateauCopy[i, j] = 0
            plateauCopy[i - couleurPiece, j + couleurPiece] = piece
            listePlateau.append(plateauCopy)

    #transformer en dame si besoin
    for p in listePlateau:
        for col in range(taillePlateau):
            if(p[0, col] == -1):
                p[0, col] = -2
            if(p[taillePlateau-1, col] == 1):
                p[taillePlateau-1, col] = 2
            
    
    return listeDeplacement, listePlateau
        
def existeCase(taillePlateau, i, j):
    if(i < taillePlateau and j < taillePlateau and i >= 0 and j >= 0):
        return True
    return False
    
            

def choixRandom(listePlateau):
    nb = random.randint(0, len(listePlateau)-1)
    return nb




def obligerManger(plateauInitial, listeDeplacement, listePlateau, taillePlateau, couleur):
    """
    on compte n pièces ennemis sur plateau initial
    si au moins un plateau a moins de n pièces ennemis
    alors on supprime tout les plateaus qui possèdent n pièces ennemis
    <=>
    comme quand le joueur joue il ne peut pas perdre ses propres pièces on peut faire np.sum(plateau)
    SI j'ai au moins un np.sum(plateau de la liste) != np.sum(plateau)
        alors je suppr tous les plateau de la liste dont sum=sum(plateau)
    """
    mangerPossible = False
    for p in listePlateau:
        if np.sum(p) != np.sum(plateauInitial):
            mangerPossible = True
            break
    
    if mangerPossible == True:
        tailleListe = len(listePlateau)
        p=0
        while(p<tailleListe):
            if np.sum(listePlateau[p]) == np.sum(plateauInitial): #on supprime tous les plateaux où on mange pas
                del(listePlateau[p])
                del(listeDeplacement[p])
                p -=  1
                tailleListe -= 1 
            p += 1
    return listeDeplacement, listePlateau

