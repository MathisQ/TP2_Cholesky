# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:44:42 2021

@author: damien
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import time
## 1 Décomposition de Cholesky

def Cholesky(A):
    """

    Paramètres
    ----------
    A : Matrice symétrique définie positive.

    Retourne
    -------
    L : Matrice de la décomposition de Cholesky

    """
    A = np.copy(A)
    c = np.shape(A)
    n = c[0]
    L = np.zeros([n, n])
    L[0,0] = m.sqrt(A[0,0])  
    for k in range(n): # On parcourt chaque colonne
        for i in range(k, n): # On parcourt les termes sous la diagonale par colonne
            if i == k: # cas d'un terme diagonale
                #print("i:", i)
                somme1 = 0
                for j in range(k):
                    #print("lkj:", L[k,j])
                    somme1 += (L[k,j])**2
                    #print("j:", j)
                    #print("k:", k)
                    #print("somme1:", somme1)
                L[k,k] = m.sqrt(A[k,k] - somme1)
                #print("Lkk:", k, k, L[k,k])
            elif i > k : # cas des termes sous la diagonale
                somme2 = 0
                for j in range(k):
                    somme2 += L[i,j]*L[k,j]
                L[i,k] = (A[i,k] - somme2) / L[k,k]
                #print("Lik:", i, k, L[i,k])
    #print(L)
    return L

## Tests
A1 = np.array([[4, -2, -4], [-2, 10, 5], [-4, 5, 6]])
B1 = np.array([[6], [-9], [-7]])
A2 = np.array([[1, 2, 4], [2, 8, 4], [4, 4, 24]])

## 2 Résolution

def ResolSystTriInf(Laug):
    n, m = np.shape(Laug)
    Ly =[0]*n
    Ly[0] = Laug[0,n]/Laug[0,0]
    for i in range(1, n):
        Ly[i] = Laug[i,n]
        for k in range(0, i):
            Ly[i] = Ly[i] - Laug[i, k] * Ly[k]
        Ly[i] =  Ly[i]/Laug[i, i]
    S = np.array([Ly])
    S = S.transpose()
    return S

def ResolSystTriSup(Taug):
    n, m = np.shape(Taug)
    L =[0]*n
    L[n-1] = Taug[n-1,n]/Taug[n-1,n-1]
    for i in range(n-2, -1, -1):
        L[i] = Taug[i,n]
        for k in range(i+1, n):
            L[i] = L[i] - Taug[i, k] * L[k]
        L[i] =  L[i]/Taug[i, i]
    S = np.array([L])
    S = S.transpose()  
    return S

def ResolCholesky(L,LT,B):
    LB = np.concatenate((L,B), axis = 1)
    y = ResolSystTriInf(LB)
    LTy = np.concatenate((LT,y), axis = 1)
    S = ResolSystTriSup(LTy) 
    return S   

def Cholesky_solution(A,B):
    L = Cholesky(A)
    LT = np.transpose(L)
    S = ResolCholesky(L, LT, B)
    return S

def linalg_Cholesky_solution(A,B):
    L = np.linalg.cholesky(A)
    LT = np.transpose(L)
    S = ResolCholesky(L, LT, B)
    return S

def ReductionGauss(A_aug):
    Aaug = np.copy(A_aug)
    n, m = np.shape(Aaug)
    for i in range(0, n-1):
        for k in range(i+1, n):
            g = Aaug[k, i] / Aaug[i, i]
            Aaug[k] = Aaug[k] - g * Aaug[i]  
    return Aaug

def Gauss(A, B):
    C = np.concatenate((A, B), axis = 1)
    T = ReductionGauss(C)
    S = ResolSystTriSup(T)
    return S

def Cholesky_alternative(A):
    A = np.copy(A)
    n, m = np.shape(A)
    L = np.eye(n) # On initialise L comme identité pour avoir les termes diagonaux .
    D = np.eye(n)
    for k in range(n):
        # D ’ abord le calcul de dkk en commençant par la somme .
        S = 0.
        for j in range(k):
            S = S + L [k, j]**2* D [j, j]
        D [k, k]= A [k, k] - S
        # Puis le calcul de la colonne k de L .
        # On calcule lik pour chaque i > k .
        for i in range(k + 1, n ):
            S = 0.
            for j in range(k):
                S = S + L[i, j]* L[k, j ] * D[j, j]
            L[i, k] = (A[i, k] - S ) / D[k, k]
    return(L, D)

def ResolSysDiag(Daug):
    n, m = np.shape(Daug)
    Ly = n*[0]
    for i in range(n):
        Ly[i] = Daug[i,-1]/Daug[i,i]
    S = np.array([Ly])
    S = S.transpose()
    return S

Daug = np.array([[2, 0, 0, 1],[0, -17, 0, 34], [0, 0, 5, 30]])

def Cholesky_alternative_solution(A,B):
    L, D = Cholesky_alternative(A)
    LT = np.transpose(L)
    LB = np.concatenate((L,B), axis = 1)
    y = ResolSystTriInf(LB)
    Dy = np.concatenate((D,y), axis = 1)
    z = ResolSysDiag(Dy)
    LTz = np.concatenate((LT, z), axis = 1)
    S = ResolSystTriSup(LTz)
    return S

A_alter = np.array([[-2, -6, 2],[-6, -17, 4], [2, 4, 5]])

## 3 Expérimentatiton

def CreationMatrice(n):
    M = np.random.rand(n, n)
    while np.linalg.det(M) == 0:
        M = np.random.rand(n, n)
    A = np.dot(np.transpose(M), M)
    B = np.random.rand(n, 1)
    return (A, B)

## Question 1

def test_temps(min, max, p):
    Liste_temps_cholesky = []
    Liste_temps_linalg = []
    Liste_temps_linalg_cholesky = []
    Liste_temps_Gauss = []
    Liste_temps_chol_alter = []
    Liste_n = []
    
    for i in range(min, max, p):
        A,B = CreationMatrice(i)
        
        temps1 = time.time()
        Cholesky_solution(A,B)
        temps2 = time.time()
        temps_cholesky = temps2 - temps1
        Liste_temps_cholesky.append(temps_cholesky)
        
        temps3 = time.time()
        np.linalg.solve(A,B)
        temps4 = time.time()
        temps_linalg = temps4 - temps3
        Liste_temps_linalg.append(temps_linalg)
        
        temps5 = time.time()
        linalg_Cholesky_solution(A,B)
        temps6 = time.time()
        temps_linalg_cholesky = temps6 - temps5
        Liste_temps_linalg_cholesky.append(temps_linalg_cholesky)
        
        temps7 = time.time()
        Gauss(A,B)
        temps8 = time.time()
        temps_Gauss = temps8 - temps7
        Liste_temps_Gauss.append(temps_Gauss)
        
        temps9 = time.time()
        Cholesky_alternative_solution(A,B)
        temps10 = time.time()
        temps_chol_alter = temps10 - temps9
        Liste_temps_chol_alter.append(temps_chol_alter)
        
        Liste_n.append(i)
    plt.figure()
    plt.plot(Liste_n, Liste_temps_cholesky, label = 'Méthode de Cholesky')
    plt.plot(Liste_n, Liste_temps_linalg, label = 'Méthode linalg.solve')
    plt.plot(Liste_n, Liste_temps_linalg_cholesky, 
             label = 'Méthode linalg.cholesky')
    plt.plot(Liste_n, Liste_temps_Gauss, label = 'Méthode de Gauss')
    plt.plot(Liste_n, Liste_temps_chol_alter, 
             label = 'Méthode de cholesky alternative')

    plt.title('Temps de calcul pour les différentes méthodes')
    plt.xlabel('Taille n de la matrice')
    plt.ylabel('Temps de calcul')
    plt.legend()
    plt.show()
    return (Liste_temps_cholesky, Liste_temps_linalg, 
            Liste_temps_linalg_cholesky, Liste_temps_Gauss,
            Liste_temps_chol_alter, Liste_n)


def test_temps_semilog(min, max, p):
    Liste_temps_cholesky = []
    Liste_temps_linalg = []
    Liste_temps_linalg_cholesky = []
    Liste_temps_Gauss = []
    Liste_temps_chol_alter = []
    Liste_n = []
    
    for i in range(min, max, p):
        A,B = CreationMatrice(i)
        
        temps1 = time.time()
        Cholesky_solution(A,B)
        temps2 = time.time()
        temps_cholesky = temps2 - temps1
        Liste_temps_cholesky.append(temps_cholesky)
        
        temps3 = time.time()
        np.linalg.solve(A,B)
        temps4 = time.time()
        temps_linalg = temps4 - temps3
        Liste_temps_linalg.append(temps_linalg)
        
        temps5 = time.time()
        linalg_Cholesky_solution(A,B)
        temps6 = time.time()
        temps_linalg_cholesky = temps6 - temps5
        Liste_temps_linalg_cholesky.append(temps_linalg_cholesky)
        
        temps7 = time.time()
        Gauss(A,B)
        temps8 = time.time()
        temps_Gauss = temps8 - temps7
        Liste_temps_Gauss.append(temps_Gauss)
        
        temps9 = time.time()
        Cholesky_alternative_solution(A,B)
        temps10 = time.time()
        temps_chol_alter = temps10 - temps9
        Liste_temps_chol_alter.append(temps_chol_alter)
        
        Liste_n.append(i)
    plt.figure()
    plt.semilogy(Liste_n, Liste_temps_cholesky, label = 'Méthode de Cholesky')
    plt.semilogy(Liste_n, Liste_temps_linalg, label = 'Méthode linalg.solve')
    plt.semilogy(Liste_n, Liste_temps_linalg_cholesky, 
                 label = 'Méthode linalg.Cholesky')
    plt.semilogy(Liste_n, Liste_temps_Gauss, label = 'Méthode de Gauss')
    plt.semilogy(Liste_n, Liste_temps_chol_alter, 
                 label = 'Méthode de Cholesky alternative')
    plt.title('Temps de calcul pour les différentes méthodes')
    plt.xlabel('Taille n de la matrice')
    plt.ylabel('Temps de calcul (ln(t))')
    plt.legend()
    plt.show()
    return (Liste_temps_cholesky, Liste_temps_linalg, 
            Liste_temps_linalg_cholesky, Liste_temps_Gauss,
            Liste_temps_chol_alter, Liste_n)

def test_temps_loglog(min, max, p):
    Liste_temps_cholesky = []
    Liste_temps_linalg = []
    Liste_temps_linalg_cholesky = []
    Liste_temps_Gauss = []
    Liste_temps_chol_alter = []
    Liste_n = []
    
    for i in range(min, max, p):
        A,B = CreationMatrice(i)
        
        temps1 = time.time()
        Cholesky_solution(A,B)
        temps2 = time.time()
        temps_cholesky = temps2 - temps1
        Liste_temps_cholesky.append(temps_cholesky)
        
        temps3 = time.time()
        np.linalg.solve(A,B)
        temps4 = time.time()
        temps_linalg = temps4 - temps3
        Liste_temps_linalg.append(temps_linalg)
        
        temps5 = time.time()
        linalg_Cholesky_solution(A,B)
        temps6 = time.time()
        temps_linalg_cholesky = temps6 - temps5
        Liste_temps_linalg_cholesky.append(temps_linalg_cholesky)
        
        temps7 = time.time()
        Gauss(A,B)
        temps8 = time.time()
        temps_Gauss = temps8 - temps7
        Liste_temps_Gauss.append(temps_Gauss)
        
        temps9 = time.time()
        Cholesky_alternative_solution(A,B)
        temps10 = time.time()
        temps_chol_alter = temps10 - temps9
        Liste_temps_chol_alter.append(temps_chol_alter)
        
        Liste_n.append(i)
    plt.figure()
    plt.loglog(Liste_n, Liste_temps_cholesky, label = 'Méthode de Cholesky')
    plt.loglog(Liste_n, Liste_temps_linalg, label = 'Méthode linalg.solve')
    plt.loglog(Liste_n, Liste_temps_linalg_cholesky, 
             label = 'Méthode linalg.cholesky')
    plt.loglog(Liste_n, Liste_temps_Gauss, label = 'Méthode de Gauss')
    plt.loglog(Liste_n, Liste_temps_chol_alter, 
             label = 'Méthode de cholesky alternative')
    plt.title('Temps de calcul pour les différentes méthodes')
    plt.xlabel('Taille n de la matrice (ln(n))')
    plt.ylabel('Temps de calcul (ln(t))')
    plt.legend()
    plt.show()
    return (Liste_temps_cholesky, Liste_temps_linalg, 
            Liste_temps_linalg_cholesky, Liste_temps_Gauss,
            Liste_temps_chol_alter, Liste_n)

## Question 2

def test_erreurs(min, max, p):
    Liste_erreur_cholesky = []
    Liste_erreur_linalg = []
    Liste_erreur_linalg_cholesky = []
    Liste_erreur_Gauss = []
    Liste_erreur_chol_alter = []
    Liste_n = []
    
    for i in range(min, max, p):
        A,B = CreationMatrice(i)
        s_chol = Cholesky_solution(A,B)
        s_linalg = np.linalg.solve(A,B)
        s_linalg_chol = linalg_Cholesky_solution(A,B)
        s_gauss = Gauss(A,B)
        s_chol_alter = Cholesky_alternative_solution(A,B)
        
        erreur_cholesky = np.linalg.norm(np.dot(A,s_chol)-B)
        erreur_linalg = np.linalg.norm(np.dot(A,s_linalg)-B)
        erreur_linalg_chol = np.linalg.norm(np.dot(A,s_linalg_chol)-B)
        erreur_gauss = np.linalg.norm(np.dot(A,s_gauss)-B)
        erreur_chol_alter = np.linalg.norm(np.dot(A,s_chol_alter)-B)

        Liste_erreur_cholesky.append(erreur_cholesky)
        Liste_erreur_linalg.append(erreur_linalg)
        Liste_erreur_linalg_cholesky.append(erreur_linalg_chol)
        Liste_erreur_Gauss.append(erreur_gauss)
        Liste_erreur_chol_alter.append(erreur_chol_alter)
        Liste_n.append(i)
    
    plt.figure()
    plt.plot(Liste_n, Liste_erreur_cholesky, label = 'Méthode de Cholesky')
    plt.plot(Liste_n, Liste_erreur_linalg, label = 'Méthode linalg.solve')
    plt.plot(Liste_n, Liste_erreur_linalg_cholesky, 
             label = 'Méthode linalg.cholesky')
    plt.plot(Liste_n, Liste_erreur_Gauss, label = 'Méthode de Gauss')
    plt.plot(Liste_n, Liste_erreur_chol_alter, 
             label = 'Méthode de cholesky alternative')

    plt.title('Erreur pour les différentes méthodes')
    plt.xlabel('Taille n de la matrice')
    plt.ylabel('Erreur ||Ax-b||')
    plt.legend()
    plt.show()
    return (Liste_erreur_cholesky, Liste_erreur_linalg, 
            Liste_erreur_linalg_cholesky, Liste_erreur_Gauss,
            Liste_erreur_chol_alter, Liste_n)
    
def test_erreurs_loglog(min, max, p):
    Liste_erreur_cholesky = []
    Liste_erreur_linalg = []
    Liste_erreur_linalg_cholesky = []
    Liste_erreur_Gauss = []
    Liste_erreur_chol_alter = []
    Liste_n = []
    
    for i in range(min, max, p):
        A,B = CreationMatrice(i)
        s_chol = Cholesky_solution(A,B)
        s_linalg = np.linalg.solve(A,B)
        s_linalg_chol = linalg_Cholesky_solution(A,B)
        s_gauss = Gauss(A,B)
        s_chol_alter = Cholesky_alternative_solution(A,B)
        
        erreur_cholesky = np.linalg.norm(np.dot(A,s_chol)-B)
        erreur_linalg = np.linalg.norm(np.dot(A,s_linalg)-B)
        erreur_linalg_chol = np.linalg.norm(np.dot(A,s_linalg_chol)-B)
        erreur_gauss = np.linalg.norm(np.dot(A,s_gauss)-B)
        erreur_chol_alter = np.linalg.norm(np.dot(A,s_chol_alter)-B)

        Liste_erreur_cholesky.append(erreur_cholesky)
        Liste_erreur_linalg.append(erreur_linalg)
        Liste_erreur_linalg_cholesky.append(erreur_linalg_chol)
        Liste_erreur_Gauss.append(erreur_gauss)
        Liste_erreur_chol_alter.append(erreur_chol_alter)
        Liste_n.append(i)
    
    plt.figure()
    plt.loglog(Liste_n, Liste_erreur_cholesky, label = 'Méthode de Cholesky')
    plt.loglog(Liste_n, Liste_erreur_linalg, label = 'Méthode linalg.solve')
    plt.loglog(Liste_n, Liste_erreur_linalg_cholesky, 
             label = 'Méthode linalg.cholesky')
    plt.loglog(Liste_n, Liste_erreur_Gauss, label = 'Méthode de Gauss')
    plt.loglog(Liste_n, Liste_erreur_chol_alter, 
             label = 'Méthode de cholesky alternative')

    plt.title('Erreur pour les différentes méthodes')
    plt.xlabel('Taille n de la matrice (ln(n))')
    plt.ylabel('Erreur ||Ax-b|| (ln||Ax-b||)')
    plt.legend()
    plt.show()
    return (Liste_erreur_cholesky, Liste_erreur_linalg, 
            Liste_erreur_linalg_cholesky, Liste_erreur_Gauss,
            Liste_erreur_chol_alter, Liste_n)