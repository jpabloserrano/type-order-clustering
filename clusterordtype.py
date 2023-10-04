import numpy as np
import random
import PyDCG
import matplotlib.pyplot as plt

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

        all_points = points + sym_points

        # Creamos otro orden con los puntos todos los puntos dados por points+sym_points

        ordered_points[tuple(x)]['all_points'] = PyDCG.geometricbasics.sort_around_point_py(x, all_points)

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
                
                # sapax = sort all points by angle from x

                sapax = ordered_points[tuple(x)]['all_points']

                a_index_p, a_index_q = sapax.index(p), sapax.index(q)

                spin = orientation(x, p, q)

                if spin == 'izq':
                    
                    if a_index_p < a_index_q:

                        distance += len(sapax[a_index_p+1:a_index_q])
                    
                    elif a_index_p > a_index_q:

                        distance += len(sapax[a_index_p+1:]) + len(sapax[2:a_index_q])

                elif spin == 'der':

                    if a_index_p > a_index_q:

                        distance += len(sapax[a_index_q+1:a_index_p])
                    
                    elif a_index_p < a_index_q:
                        
                        distance += len(sapax[2:a_index_p]) + len(sapax[a_index_q+1:])

        return int(distance / 2)

# Función para calcular el centroide de un conjunto de puntos respecto a una distancia cualquiera.
# En esta función se utiliza la función definida para este problema (delta), pero se espera que funcione para cualquier distancia.
# El centroide se calcula como aquel punto p que minimice la suma de distancias \sum delta(p,q_{i}).
# Este método es O(kn^{2}) donde k es tamaño del cluster.

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
# Este método es O(n^{2}\log n) + O(k^{2}n^{2}).

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

            # El tiempo de ejecución de la siguiente línea es O(kn^{2})

            distances = [delta(p=p, q=center, points=points, ordered_points=op) for center in centers]

            # Encontramos la distancia de p a un centro mínima.
            
            min_index = np.argmin(distances)

            # Asignación del punto p al cluster "min_index"

            cluster[min_index].append(p)

        # Cálculo de nuevos centros

        updated_centers = list()

        for j in cluster.keys():

            # Se añade el centroide (respecto a "delta") al nuevo conjunto de centros

            # El tiempo de ejecución de la siguiente línea es O(k^{2}n^{2})
            
            updated_centers.append(centroid(cluster=cluster[j], ordered_points=op))
            
        f += 1

        # Condición de Lloyd cuando los centros ya no se mueven
        
        if np.allclose(a=updated_centers, b=centers): break

        centers = updated_centers
    
    print(f'\n{f} iteraciones.')
    
    return {'cluster': cluster, 'centers': centers}

# Lectura de puntos

points = vectors(path=r'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\puntos.txt')

for k in [3,4,6]:

    clustering = order_type_clustering(points=points, k=k)

    for i in range(k):

        cluster = clustering['cluster'][i]

        # Desempaquetado de las coordenadas en listas separadas
        xcoordinate, ycoordinate = zip(*cluster)

        #color = colors[i % len(colors)]

        # Crea un gráfico de dispersión
        plt.scatter(xcoordinate, ycoordinate, label=f"Cluster {i+1}")

    # Agrega una leyenda
    plt.legend()

    # Muestra el gráfico
    plt.show()