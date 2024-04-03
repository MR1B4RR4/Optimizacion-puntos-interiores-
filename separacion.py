#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:39:32 2024

@author: santiagocasanova
"""

# Samuel Elí Méndez Sánchez 196659
# Santiago Casanova Díaz 189756
# Eduardo Mateo Guajardo Rodriguez 166273
# Sebastian Ibarra Del Corral 193992
# Sandra Reyes Benavides 179149

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import qp_intpoint as qpi

iris = pd.read_csv("Iris.csv")

n = 149
x = iris["SepalLengthCm"]
y = iris["SepalWidthCm"]
z = iris["PetalLengthCm"]
c = iris["PetalWidthCm"]
species = iris["Species"]

#%%

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=c, cmap='viridis')

plt.colorbar(scatter, label='PetalWidthCm')

ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('PetalLenghtCm')

for i in range(n):
    if(np.random.rand()<0.1):
        ax.text(x[i], y[i], z[i], species[i], color='black', fontsize=8)

plt.show()

#%%

A = iris[iris["Species"]=="Iris-setosa"]
B = iris[iris["Species"]=="Iris-virginica"]
C = iris[iris["Species"]=="Iris-versicolor"]

A_mat = A[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
B_mat = B[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
C_mat = C[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Vec_m1= np.full((50, 1), -1)
Vec_1 = np.full((50, 1), 1)

d = np.full((100, 1), 1)
c = np.full((5,1),0)

# Caso Setosa Virginica

concatenated_matrix_up = np.hstack((-A_mat, Vec_m1))
concatenated_matrix_down = np.hstack((B_mat, Vec_1))
final_matrix = np.vstack((concatenated_matrix_up,concatenated_matrix_down))
F = final_matrix

Q=np.eye(5)
F=final_matrix

(x,mu,z,iter)= qpi.myqp_intpoint_modificado(Q, F, c, d)

print(x)

#%%

w = x[:-1]  
beta = x[-1] 
A = A_mat
B = B_mat

# Calcula ATw - e y BTw - e
e = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e * beta

e = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e * beta

# Graficación
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(ATw_minus_e, label='ATw - e')
plt.title('Conjunto A')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.subplot(1, 2, 2)
plt.plot(BTw_minus_e, label='BTw - e')
plt.title('Conjunto B')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.tight_layout()
plt.show()

#%%

# Asumiendo que w, beta, A_mat, y B_mat están definidos
w = x[:-1]  # vector w
beta = x[-1]  # valor beta
A = A_mat  # matriz A
B = B_mat  # matriz B

# Calcula ATw - e y BTw - e
e_A = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e_A * beta

e_B = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e_B * beta

# Asegurémonos de que las dimensiones de ATw_minus_e y BTw_minus_e son adecuadas para graficar
# Por claridad, vamos a aplanar los arrays para garantizar que son 1D
ATw_minus_e = ATw_minus_e.flatten()
BTw_minus_e = BTw_minus_e.flatten()

# Graficación en una sola gráfica usando scatter
plt.figure(figsize=(10, 5))

# Aquí no necesitamos generar índices ya que scatter puede manejar los valores directamente
plt.scatter(np.arange(len(ATw_minus_e)), ATw_minus_e, color='blue', label='Conjunto A', alpha=0.7)
plt.scatter(np.arange(len(BTw_minus_e)) + len(ATw_minus_e), BTw_minus_e, color='red', label='Conjunto B', alpha=0.7)

plt.title('Comparación de Conjunto A y B respecto al Hiperplano')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor (ATw - e, BTw - e)')
plt.legend()

plt.tight_layout()
plt.show()

#%%


A = A_mat
B = B_mat

w = x[:-1]  
beta = x[-1] 


# Definir las combinaciones de características
combinaciones = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Crear una figura y ejes para las subgráficas
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparación de Grupos A y B en diferentes combinaciones de características')

for i, (feat1, feat2) in enumerate(combinaciones):
    ax = axes[i//3, i%3]
    # Graficar los puntos de A y B para cada combinación de características
    ax.scatter(A[:, feat1], A[:, feat2], color='blue', label='Grupo A')
    ax.scatter(B[:, feat1], B[:, feat2], color='red', label='Grupo B')
    ax.set_xlabel(f'Característica {feat1+1}')
    ax.set_ylabel(f'Característica {feat2+1}')
    ax.legend()


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%

# Caso Versicolor Virginica


concatenated_matrix_up = np.hstack((-A_mat, Vec_m1))
concatenated_matrix_down = np.hstack((C_mat, Vec_1))
final_matrix = np.vstack((concatenated_matrix_up,concatenated_matrix_down))
F2 = final_matrix

(x,mu,z,iter)= qpi.myqp_intpoint_modificado(Q, F2, c, d)

print(x)

#%%

w = x[:-1]  
beta = x[-1] 
A = A_mat
B = C_mat

# Calcula ATw - e y BTw - e
e = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e * beta

e = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e * beta

# Graficación
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(ATw_minus_e, label='ATw - e')
plt.title('Conjunto A')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.subplot(1, 2, 2)
plt.plot(BTw_minus_e, label='BTw - e')
plt.title('Conjunto C')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.tight_layout()
plt.show()

#%%
w = x[:-1]  
beta = x[-1] 
A = A_mat
B = C_mat

# Calcula ATw - e y BTw - e
e_A = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e_A * beta

e_B = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e_B * beta

# Asegurémonos de que las dimensiones de ATw_minus_e y BTw_minus_e son adecuadas para graficar
# Por claridad, vamos a aplanar los arrays para garantizar que son 1D
ATw_minus_e = ATw_minus_e.flatten()
BTw_minus_e = BTw_minus_e.flatten()

# Graficación en una sola gráfica usando scatter
plt.figure(figsize=(10, 5))

# Aquí no necesitamos generar índices ya que scatter puede manejar los valores directamente
plt.scatter(np.arange(len(ATw_minus_e)), ATw_minus_e, color='blue', label='Conjunto A', alpha=0.7)
plt.scatter(np.arange(len(BTw_minus_e)) + len(ATw_minus_e), BTw_minus_e, color='red', label='Conjunto C', alpha=0.7)

plt.title('Comparación de Conjunto A y C respecto al Hiperplano')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor (ATw - e, CTw - e)')
plt.legend()

plt.tight_layout()
plt.show()

#%%

A = A_mat
B = C_mat

w = x[:-1]  
beta = x[-1] 


# Definir las combinaciones de características
combinaciones = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Crear una figura y ejes para las subgráficas
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparación de Grupos A y C en diferentes combinaciones de características')

for i, (feat1, feat2) in enumerate(combinaciones):
    ax = axes[i//3, i%3]
    # Graficar los puntos de A y B para cada combinación de características
    ax.scatter(A[:, feat1], A[:, feat2], color='blue', label='Grupo A')
    ax.scatter(B[:, feat1], B[:, feat2], color='red', label='Grupo C')
    ax.set_xlabel(f'Característica {feat1+1}')
    ax.set_ylabel(f'Característica {feat2+1}')
    ax.legend()


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%

# Caso Versicolor Setosa

concatenated_matrix_up = np.hstack((-B_mat, Vec_m1))
concatenated_matrix_down = np.hstack((C_mat, Vec_1))
final_matrix = np.vstack((concatenated_matrix_up,concatenated_matrix_down))
F3 = final_matrix

(x,mu,z,iter)= qpi.myqp_intpoint_modificado(Q, F3, c, d)


print(x)

#%%

w = x[:-1]  
beta = x[-1] 
A = C_mat
B = A_mat

# Calcula ATw - e y BTw - e
e = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e * beta

e = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e * beta

# Graficación
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(ATw_minus_e, label='ATw - e')
plt.title('Conjunto B')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.subplot(1, 2, 2)
plt.plot(BTw_minus_e, label='BTw - e')
plt.title('Conjunto C')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor')
#plt.legend()

plt.tight_layout()
plt.show()


#%%

w = x[:-1]  
beta = x[-1] 
A = B_mat
B = C_mat

# Calcula ATw - e y BTw - e
e_A = np.ones(A.shape[0])
ATw_minus_e = A.dot(w) - e_A * beta

e_B = np.ones(B.shape[0])
BTw_minus_e = B.dot(w) - e_B * beta

# Asegurémonos de que las dimensiones de ATw_minus_e y BTw_minus_e son adecuadas para graficar
# Por claridad, vamos a aplanar los arrays para garantizar que son 1D
ATw_minus_e = ATw_minus_e.flatten()
BTw_minus_e = BTw_minus_e.flatten()

# Graficación en una sola gráfica usando scatter
plt.figure(figsize=(10, 5))

# Aquí no necesitamos generar índices ya que scatter puede manejar los valores directamente
plt.scatter(np.arange(len(ATw_minus_e)), ATw_minus_e, color='blue', label='Conjunto B', alpha=0.7)
plt.scatter(np.arange(len(BTw_minus_e)) + len(ATw_minus_e), BTw_minus_e, color='red', label='Conjunto C', alpha=0.7)

plt.title('Comparación de Conjunto B y C respecto al Hiperplano')
plt.xlabel('Índice del Punto')
plt.ylabel('Valor (BTw - e, CTw - e)')
plt.legend()

plt.tight_layout()
plt.show()


#%%

A = B_mat
B = C_mat

w = x[:-1]  
beta = x[-1] 


# Definir las combinaciones de características
combinaciones = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Crear una figura y ejes para las subgráficas
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparación de Grupos B y C en diferentes combinaciones de características')

for i, (feat1, feat2) in enumerate(combinaciones):
    ax = axes[i//3, i%3]
    # Graficar los puntos de A y B para cada combinación de características
    ax.scatter(A[:, feat1], A[:, feat2], color='blue', label='Grupo B')
    ax.scatter(B[:, feat1], B[:, feat2], color='red', label='Grupo C')
    ax.set_xlabel(f'Característica {feat1+1}')
    ax.set_ylabel(f'Característica {feat2+1}')
    ax.legend()


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

