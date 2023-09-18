from PyDCG import geometricbasics
import numpy as np
import random

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

        # Creación del orden respecto a x de los puntos "originales"

        ordered_points[tuple(x)]['ordered_points'] = geometricbasics.sort_around_point_py(x, points)

        # Creamos otro orden con los puntos simétricos dados por sym_points

        ordered_points[tuple(x)]['symmetric_points'] = geometricbasics.sort_around_point_py(x, sym_points)
    
    return ordered_points

# Función para el cálculo de la distancia en "tipo de orden".
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
                
                sym_p = [(2 * x[0]) - p[0], (2 * x[1]) - p[1]]

                sym_q = [(2 * x[0]) - q[0], (2 * x[1]) - q[1]]

                # spax = sorted points by angle from x
                
                spax = ordered_points[tuple(x)]['ordered_points']
                
                sym_spax = ordered_points[tuple(x)]['symmetric_points']
                
                index_p, index_q = spax.index(p), spax.index(q) # Esta línea se puede hacer en O(log n)
                
                spin = orientation(x, p, q)

                if spin == 'izq': distance += (len(spax[index_p+1:index_q]) + len(sym_spax[index_p+1:index_q])) / 2

                elif spin == 'der': distance += (len(spax[index_q+1:index_p]) + len(sym_spax[index_q+1:index_p])) / 2
                
        return int(distance / 2)

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

# Ejemplo de uso

p = [3,3]

q = [0, 2]

points = [[0, 3], [4, 0], p, q]

result = delta(p, q, points, ordered_points(points))

print(f"Distancia entre {p} y {q}: {result}")
