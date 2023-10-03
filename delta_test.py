from PyDCG import geometricbasics
import numpy as np
import random
import PyDCG

# Función para conocer la orientacióno de 3 puntos A, B, C

def orientation(A: list, B: list, C: list) -> str:
    
    # Función para calcular la orientación entre tres puntos
    
    det = ((B[0]-A[0])*C[1])+((A[0]-C[0])*B[1])+((C[0]-B[0])*A[1])
    
    if det == 0:
    
        return 'col'  # Colineales
    
    return 'izq' if det > 0 else 'der'  # En sentido horario o antihorario

# Método para realizar el ordenamiento respecto a ángulo de todos los puntos.
# Devuelve un diccionario cuyas claves son las tuplas de los vectores para acceder al orden respecto al ángulo.
# Este método es O(n^{2}\log n).

def ordered_points(points: list) -> dict:

    # Uso de un diccionario para poder accesar al orden de cada punto x así como los ángulos respecto a x.

    ordered_points = dict()

    for x in points:

        ordered_points[tuple(x)] = dict()

        sym_points = list()

        for p in points:
                
            sym_p = [(2 * x[0]) - p[0], (2 * x[1]) - p[1]]

            sym_points.append(sym_p)

        all_points = points + sym_points

        # Creación del orden respecto a x de los puntos "originales"

        ordered_points[tuple(x)]['ordered_points'] = geometricbasics.sort_around_point_py(x, points)

        # Creamos otro orden con los puntos simétricos dados por sym_points

        ordered_points[tuple(x)]['symmetric_points'] = geometricbasics.sort_around_point_py(x, sym_points)

        # Creamos otro orden con los puntos todos los puntos dados por points+sym_points

        ordered_points[tuple(x)]['all_points'] = geometricbasics.sort_around_point_py(x, all_points)

    return ordered_points

# Función para el cálculo de la distancia en "tipo de orden".
# Este método es O(n\log n).

def delta(p: list, q: list, points: list, ordered_points: dict, debug=False) -> int:

    # Condición de distancia para puntos fuera de "points"

    if (q not in points) or (p not in points):

        return 0

    else:

        # Inicialización de la distancia.

        distance = 0

        # Lista que consiste en los puntos distintos a p y q.

        not_p_not_q = [point for point in points if point != p and point != q]

        for x in not_p_not_q:
                
                # spax = sorted points by angle from x
                if debug: print(f'Center: {x}\n')

                sym_spax = ordered_points[tuple(x)]['symmetric_points']
                
                if debug: print(f'Symmetric Points: {sym_spax}\n')

                sapax = ordered_points[tuple(x)]['all_points']
                if debug: print(f'All points: {sapax}\n')
                a_index_p, a_index_q = sapax.index(p), sapax.index(q)
                if debug: print(f'All points index: {(a_index_p,a_index_q)}\n')
                spin = orientation(x, p, q)
                if debug: print(f'Orientation: {spin}\n')

                if spin == 'izq':
                    
                    if a_index_p < a_index_q:

                        distance += len(sapax[a_index_p+1:a_index_q])
                    
                        #distance += len(spax[index_p+1:index_q]) + len(spax[sym_index_q+1:sym_index_p])

                    elif a_index_p > a_index_q:

                        distance += len(sapax[a_index_p+1:]) + len(sapax[2:a_index_q])

                        #distance += len(spax[index_p+1:]) + len(spax[1:index_q]) + len(spax[sym_index_q+1:]) + len(spax[1:sym_index_p])

                elif spin == 'der':

                    if a_index_p > a_index_q:

                        distance += len(sapax[a_index_q+1:a_index_p])
                    
                        #distance += len(spax[index_q+1:index_p]) + len(spax[sym_index_p+1:sym_index_q])

                    elif a_index_p < a_index_q:
                        
                        distance += len(sapax[2:a_index_p]) + len(sapax[a_index_q+1:])
                        #distance += len(spax[index_q+1:]) + len(spax[1:index_p]) + len(spax[sym_index_p+1:]) + len(spax[1:sym_index_q])
                if debug: print(f'Distance: {distance}\n')
        return int(distance / 2)
    
# Función para verificar si la línea entre X e Y cruza el segmento de P a Q
def is_crossing(P: list, Q: list, X: list, Y: list) -> bool:

    # Los puntos son colineales o los segmentos se cruzan

    return orientation(X, Y, P) != orientation(X, Y, Q)
    
#Función para contar el par de puntos que cruzan a P y Q.
def count_crossing_pairs(P: list, Q: list, points: list) -> int:
    
    count = 0

    # Iterar sobre cada par de puntos

    not_P_not_Q = [point for point in points if point != P and point != Q]

    for i in range(len(not_P_not_Q)):

        for j in range(i + 1, len(not_P_not_Q)):

            if is_crossing(P, Q, not_P_not_Q[i], not_P_not_Q[j]):
    
                count += 1

    return count


for _ in range(100):
    # Genera un número aleatorio entre 5 y 20 para determinar la cantidad de listas
    numero_de_listas = random.randint(5, 10)

    # Lista que almacenará las listas de vectores
    points = []

    # Genera las listas de vectores aleatorios
    for _ in range(numero_de_listas):
        # Genera un número aleatorio entre -20 y 100 para la coordenada x
        x = random.randint(-20, 100)
        # Genera un número aleatorio entre -20 y 100 para la coordenada y
        y = random.randint(-20, 100)
        # Crea una lista que representa el vector con coordenadas aleatorias
        vector = [x, y]
        # Agrega el vector a la lista de vectores
        points.append(vector)

    # Elige aleatoriamente dos puntos p y q de la lista de vectores
    p = random.choice(points)
    q = random.choice(points)

    fast = delta(p,q,points,ordered_points(points))
    slow = count_crossing_pairs(p,q,points)

    if fast != slow:

        print(f'Fast: {fast}')
        print(f'Slow: {slow}')

        print(p)
        print(q)
        print(points)

        delta(p,q,points,ordered_points(points), debug=True)

        p.append(0)
        q.append(0)
        L=[]

        for i in range(len(points)):
            for j in range(i+1,len(points)):
                L.append(PyDCG.line.Line(p=points[i],q=points[j]))

        vis=PyDCG.visualizer2.Vis(points=points,lines=L)
        break