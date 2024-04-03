# Función myqp_intpoint_modificado
La función myqp_intpoint_modificado implementa un método modificado del punto interior para resolver problemas de programación cuadrática. Utiliza un enfoque iterativo para encontrar el mínimo de una función cuadrática sujeta a restricciones de igualdad lineales.

### Dependencias
Esta función requiere la biblioteca numpy para su correcta ejecución. Asegúrate de tenerla instalada en tu entorno de trabajo.

### Parámetros
Q: Matriz cuadrada numpy de dimensión 
�
×
�
n×n, donde n es el número de variables de decisión. Esta matriz representa la parte cuadrática del objetivo.
F: Matriz numpy de dimensión 
�
×
�
p×n, donde p es el número de restricciones. Representa las restricciones de igualdad.
c: Vector columna numpy de dimensión 
�
×
1
n×1. Representa el término lineal del objetivo.
d: Vector columna numpy de dimensión 
�
×
1
p×1. Representa el lado derecho de las restricciones de igualdad.
### Salida
La función retorna una tupla con los siguientes elementos:

x: Vector columna numpy de dimensión 
�
×
1
n×1. Representa la solución óptima de las variables de decisión.
mu: Vector columna numpy de dimensión 
�
×
1
p×1. Representa los multiplicadores de Lagrange asociados a las restricciones de igualdad.
z: Vector columna numpy de dimensión 
�
×
1
p×1. Representa las variables de holgura asociadas a las restricciones de igualdad.
iter: Entero que indica el número de iteraciones realizadas hasta alcanzar la solución.
