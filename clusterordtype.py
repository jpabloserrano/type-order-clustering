from PyDCG import geometricbasics
import numpy as np
import random
import math


# Función para leer un archivo txt (la entrada de la función es la ruta al archivo)
# para almacenar la lista de los vectores del archivo txt.

def vectors(path: str) -> list:

    points = list()

    with open(path, 'r') as file:

        for line in file:

            # Lee cada línea del archivo y divide los números usando split()

            a, b = map(float, line.strip().split())

            points.append([int(a),int(b)])
            
    return points

# Método para realizar el ordenamiento respecto a ángulo de todos los puntos.
# Devuelve un diccionario cuyas claves son las tuplas de los vectores para acceder al orden respecto al ángulo.
# Este método es O(n^{2}\log n).

def ordered_points(points: list) -> dict:

    # Uso de un diccionario para poder accesar al orden de cada punto x así como los ángulos respecto a x.

    ordered_points = dict()

    for x in points:

        # Definimos un diccionario que va a contener los puntos ordenados y los ángulos respectivos.

        ordered_points[tuple(x)] = dict()

        # sapp = sort around points by

        sapp = geometricbasics.sort_around_point_py(p=x, points=points)

        ordered_points[tuple(x)]['ordered_points'] = sapp

        angles = list()

        for p in sapp:
            
            # Calcula el ángulo en radianes entre p y x.
            
            angle = math.atan2(p[1], p[0]) - math.atan2(x[1], x[0])

            # Ángulo esté en el rango [0, 2*pi]
            
            if angle < 0:
            
                angle += 2 * math.pi
            
            angles.append(angle)

        ordered_points[tuple(x)]['angles'] = angles

    
    return ordered_points

# Método para encontrar el número de ángulos en una lista de ángulos entre angle_mix y max_angle
# Este método es log(n) utilizando búsqueda binaria.
# Este método está hecho para una lista angles ordenada en forma decreciente

def points_between_angles(angles: list, min_angle: float, max_angle: float) -> int:
    
    # Búsqueda binaria para encontrar el índice del primer elemento menor que min_angle
    
    left = 0
    
    right = len(angles) - 1
    
    while left <= right:
    
        mid = left + (right - left) // 2
    
        if angles[mid] >= min_angle:
    
            left = mid + 1
    
        else:
    
            right = mid - 1
    

    # geq contiene el índice del primer elemento mayor que min_angle
    
    geq = left

    # Búsqueda binaria para encontrar el índice del último elemento menor que max_angle
    
    left = 0
    
    right = len(angles) - 1
    
    while left <= right:
    
        mid = left + (right - left) // 2
    
        if angles[mid] > max_angle:
    
            left = mid + 1
    
        else:
    
            right = mid - 1
    
    # leq contiene el índice del último elemento menor que max_angle

    leq = right

    return geq-leq-1

# Función para el cálculo de la distancia en "tipo de orden".
# La distancia entre dos puntos "p" y "q" de un conjunto de puntos "points"
# se calcula como el número de par de puntos cuya recta que los une intersecta
# al segmento de recta que une a p con q.
# Este método es O(n\log n).

def delta(p: list, q: list, points: list, ordered_points: dict) -> int:

    # Condición de distancia para puntos fuera de "points"

    if (q not in points) or (p not in points):

        return 0

    else:

        # Inicialización de la distancia.

        distance = 0

        # Lista que consiste en los puntos distintos a p y q.

        not_p_not_q = [point for point in points if point != p and point != q]

        for x in not_p_not_q:

                # spa = sorted points by angle from x

                sapp = ordered_points[tuple(x)]['ordered_points']

                angles = ordered_points[tuple(x)]['angles']

                # Encontramos los índices del punto de mayor ángulo y menor ángulo respectivamente.

                max_angle, min_angle = min(sapp.index(p), sapp.index(q)), max(sapp.index(p), sapp.index(q))

                # Incremento de la distancia en el número de puntos que estén en la region que determinan las rectas que unen a p con x
                # y q con x.
                # Esos puntos seran aquellos cuyos ángulos estén entre el mínimo y el máximo y min-180 y max-180
                
                distance += points_between_angles(angles=angles, min_angle=angles[min_angle]-math.pi, max_angle=angles[max_angle]-math.pi)
                
                distance += points_between_angles(angles=angles, min_angle=angles[min_angle], max_angle=angles[max_angle])

        return distance

# Función para calcular el centroide de un conjunto de puntos respecto a una distancia cualquiera.
# En esta función se utiliza la función definida para este problema (delta), pero se espera que funcione para cualquier distancia.
# El centroide se calcula como aquel punto p que minimice la suma de distancias \sum delta(p,q_{i}).
# Este método es O(kn^{2}\log n) donde k=tamaño del cluster.

def centroid(cluster: list, ordered_points: list) -> list:

    # La entrada (cluster) se refiere a la agrupación de elementos donde se desea calcular el centroide.

    # Verificamos que el cluster sea no vacío en caso contrario el centroide se define como el vector (0,0)

    if cluster:
     
        # Inicialización de la distancia mínima 

        min_distance = float('inf')

        centroid = None

        for p in cluster:

            sum_distance = sum([delta(p=p, q=q, points=cluster, ordered_points=ordered_points) for q in cluster])

            if sum_distance < min_distance:
                
                min_distance = sum_distance

                centroid = p

        return centroid
    
    else:

        return [0,0]

# Algoritmo de Lloyd. Devuelve un diccionario para representar los clusters y los centros.
# Este método es O(n^{2}\log n).

def order_type_clustering(points: list, k: int) -> dict:

    # op = ordered points

    op = ordered_points(points=points)
    
    # Elección aleatoria de k centros

    centers = random.sample(points, k)

    # Variable indicadora para el número de iteraciones.

    f = 1

    while True:
    
        # Diccionario para representar los clusters

        cluster = dict()


        for j in range(k): cluster[j] = list()
        
        for p in points:

            # Distancia del punto p a cada centro. 

            # Este tiempo de ejecución es O(kn\log(n))

            distances = [delta(p=p, q=center, points=points, ordered_points=op) for center in centers]

            # Encontramos la distancia de p a un centro mínima.
            
            min_index = np.argmin(distances)

            # Asignación del punto p al cluster "min_index"

            cluster[min_index].append(p)

        # Cálculo de nuevos centros

        updated_centers = list()

        for j in cluster.keys():

            # Se añade el centroide (respecto a "delta") al nuevo conjunto de centros
            
            updated_centers.append(centroid(cluster=cluster[j]))
            
        f += 1

        # Condición de Lloyd cuando los centros ya no se mueven
        
        if np.allclose(a=updated_centers, b=centers): break

        centers = updated_centers
    
    print(f'\n{f} iteraciones.')
    
    return {'cluster': cluster, 'centers': centers}