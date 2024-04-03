#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:41:18 2024

@author: santiagocasanova
"""

# Samuel Elí Méndez Sánchez 196659
# Santiago Casanova Díaz 189756
# Eduardo Mateo Guajardo Rodriguez 166273
# Sebastian Ibarra
# Sandra Reyes Benavides 179149

import numpy as np

def paso_intpoint(u, v):
    p = len(u)
    v_alfa = np.ones(p)
    for i in range(p):
         if v[i] < 0:
             v_alfa[i] = -u[i] / v[i]
    alfa = min(np.amin(v_alfa), 1.0)        
    return alfa

def myqp_intpoint_modificado(Q, F, c, d):
    n = len(c)
    p = len(d)
    tol = 10**(-5)
    maxiter = 100
    iter = 0
    x = np.ones((n, 1)) # Asegúrate de que x es un vector columna
    mu = np.ones((p, 1)) # Asegúrate de que mu es un vector columna
    z = np.ones((p, 1)) # Asegúrate de que z es un vector columna
    
    while iter < maxiter:
        iter += 1
        # Calcula los componentes de cnpo para esta iteración
        v1 = np.dot(Q, x) - np.dot(F.T, mu) + c
        v3 = -np.dot(F, x) + z + d
        v4 = mu * z # Multiplicación elemento a elemento

        # Usa np.vstack para concatenar verticalmente y formar cnpo correctamente
        cnpo = np.vstack([v1, v3, v4])
        norma_cnpo = np.linalg.norm(cnpo)
        
        if norma_cnpo <= tol:
            break
        
        # Ajustes para preparar la solución del sistema lineal
        M = np.zeros((n + 2*p, n + 2*p))
        M[:n, :n] = Q
        M[:n, n:n+p] = -F.T
        M[n:n+p, :n] = -F
        M[n:n+p, n+p:] = np.eye(p)
        M[n+p:, n:n+p] = np.diagflat(z)
        M[n+p:, n+p:] = np.diagflat(mu)

        # Ajuste el lado derecho del sistema
        cnpo_pert = cnpo.copy()
        cnpo_pert[n+p:] -= 0.5 / p * np.sum(mu * z)
        
        # Resuelve el sistema lineal
        dw = np.linalg.solve(M, -cnpo_pert)
        dx = dw[:n]
        dmu = dw[n:n+p]
        dz = dw[n+p:]
        
        # Calcula los pasos para mu y z
        alfa1 = paso_intpoint(mu.flatten(), dmu.flatten())
        alfa2 = paso_intpoint(z.flatten(), dz.flatten())
        alfa = 0.95 * min(alfa1, alfa2, 1.0)

        # Actualiza las variables
        x += alfa * dx
        mu += alfa * dmu
        z += alfa * dz

        print("Iteración:", iter, "| Norma de cnpo:", norma_cnpo)

    return x, mu, z, iter


