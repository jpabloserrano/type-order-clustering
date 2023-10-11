import json
import numpy as np
import random
import PyDCG
import matplotlib.pyplot as plt
import pickle

def order_type_clustering(points: list, k: int) -> dict:

    def orientation(A: list, B: list, C: list) -> str:
        # Función para conocer la orientacióno de 3 puntos A, B, C
        
        # Función para calcular la orientación entre tres puntos
        
        det = ((B[0]-A[0])*C[1])+((A[0]-C[0])*B[1])+((C[0]-B[0])*A[1])
        
        if det == 0:
        
            return 'col'  # Colineales
        
        return 'izq' if det > 0 else 'der'  # En sentido horario o antihorario

    def polar_binary_search(points, p):

        # Realiza una búsqueda binaria en la lista ordenada de puntos en sentido polar en O(log n)
        
        left, right = 0, len(points) - 1
        
        while left <= right:
            
            mid = left + (right - left) // 2
            
            if points[mid] == p:
                
                return mid
            
            elif orientation(points[0], points[mid], p) == 'der':
                
                right = mid -1
            
            else: 
            
                left = mid + 1
        
        return -1  # El punto no se encontró en la lista

    def ordered_points(points: list) -> dict:

        # Método para realizar el ordenamiento respecto a ángulo de todos los puntos.
        # Devuelve un diccionario cuyas claves son las tuplas de los vectores para acceder al orden respecto al ángulo.
        # Este método es O(n^{2}\log n).

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

    def delta(p: list, q: list, points: list, ordered_points: dict) -> int:
        
        # Función para el cálculo de la distancia en "tipo de orden".
        # Este método es O(n\log n).

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

                    a_index_p, a_index_q = polar_binary_search(sapax, p), polar_binary_search(sapax, q)

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
    
    def centroid(cluster: list) -> list:

        # Función para calcular el centroide de un conjunto de puntos respecto a una distancia cualquiera.
        # En esta función se utiliza la función definida para este problema (delta), pero se espera que funcione para cualquier distancia.
        # El centroide se calcula como aquel punto p que minimice la suma de distancias \sum delta(p,q_{i}).
        # Este método es O(n^{2}log n) donde k es tamaño del cluster.

        # La entrada (cluster) se refiere a la agrupación de elementos donde se desea calcular el centroide.
        # Verificamos que el cluster sea no vacío en caso contrario el centroide se define como el vector (0,0)

        op = ordered_points(points=cluster)

        if cluster:
        
            # Inicialización de la distancia mínima 

            min_distance = float('inf')

            centroid = None

            for p in cluster:

                sum_distance = sum([delta(p=p, q=q, points=cluster, ordered_points=op) for q in cluster])

                if sum_distance < min_distance:
                    
                    min_distance = sum_distance

                    centroid = p

            return centroid
        
        else:

            return [0,0]

    # Algoritmo de Lloyd. Devuelve un diccionario para representar los clusters y los centros.

    # op = ordered points (O(n^{2}log n))

    op = ordered_points(points=points)
    
    # Elección aleatoria de k centros

    centers = random.sample(points, k)

    # Variable indicadora para el número de iteraciones.

    f = 1

    while True:

        # Diccionario para representar los clusters

        updated_cluster = dict()

        for j in range(k): updated_cluster[j] = list()

        # Tiempo del siguiente ciclo: O(kn^{2}\log n)
        
        for p in points:

            # Distancia del punto p a cada centro.

            distances = [delta(p=p, q=center, points=points, ordered_points=op) for center in centers]

            # Encontramos la distancia de p a un centro mínima.
            
            min_distance_index = np.argmin(distances)

            # Asignación del punto p al cluster "min_distance_index"

            updated_cluster[min_distance_index].append(p)

        # Cálculo de nuevos centros

        updated_centers = list()

        # Tiempo del siguiente ciclo: O()

        for j in range(k):

            # Se añade el centroide (respecto a "delta") al nuevo conjunto de centros

            updated_centers.append(centroid(cluster=updated_cluster[j]))
            
        f += 1
        
        if centers == updated_centers: break
        
        else: centers = updated_centers


    print(f'{f} iteraciones.\n')
    
    return {'cluster': updated_cluster, 'centers': centers}

# Lectura de puntos

file=open(r'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\rectilinear_crossing_number.pkl','rb')

D=pickle.load(file, encoding='latin1')

points = D[1500]['pts']

k = 6

clustering = order_type_clustering(points=points, k=k)

# Guarda el diccionario en un archivo de texto como JSON
with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(len(points))+'pts_'+str(k)+'clusters.txt', "w") as f:
    
    json.dump(clustering, f)

for i in range(k):

    cluster = clustering['cluster'][i]

    try: pts = D[len(cluster)]['pts']
    except: continue

    print(f'Num. Puntos Cluster {i+1}: {len(cluster)}')
    print(f'Crossing Number Cluster {i+1}: {PyDCG.crossing.count_crossings_py(cluster)}')
    print(f'Optimal (until now) Crossing Number {i+1}: {PyDCG.crossing.count_crossings_py(pts)}\n')


    # Desempaquetado de las coordenadas en listas separadas
    xcoordinate, ycoordinate = zip(*cluster)

    # Crea un gráfico de dispersión
    plt.scatter(xcoordinate, ycoordinate, label=f"Cluster {i+1}")

# Agrega una leyenda
plt.legend()

# Agrega una leyenda
plt.savefig(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(k)+'pts\\'+str(len(points))+'pts_'+str(k)+'clusters.png')

# Muestra el gráfico
plt.show()