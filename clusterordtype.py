import json
import os
import numpy as np
import random
import PyDCG
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def order_type_clustering(points: list, k: int, plot=False) -> dict:

    def orientation(A: list, B: list, C: list) -> str:

        ''' 
            Función para conocer la orientacióno de 3 puntos A, B, C

            return: izq->izquierda, der->derecha, col->colineales
        
        '''
        
        det = ((B[0]-A[0])*C[1])+((A[0]-C[0])*B[1])+((C[0]-B[0])*A[1])
        
        if det == 0:
        
            return 'col'  # Colineales
        
        return 'izq' if det > 0 else 'der'  # En sentido horario o antihorario

    def polar_binary_search(points: list, p: list) -> int:
        
        '''
            Realiza una búsqueda binaria en la lista ordenada de puntos en sentido polar en O(log n)
            -Parametrs:
            points: muestra de puntos
            p: punto a buscar

            return: índice donde se encuentra p.
        
        '''
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
        
        '''
            
            Método para realizar el ordenamiento respecto a ángulo de todos los puntos.
            Devuelve un diccionario cuyas claves son las tuplas de los vectores para acceder al orden respecto al ángulo.
            Este método es O(n^{2}\log n).

            Uso de un diccionario para poder accesar al orden de cada punto x así como los ángulos respecto a x.

            -Parametros:
            points: muestra de puntos donde se hará el ordenamiento polar.

            -return:
            Regresa un diccionario con cada orden a partir de cada punto x. Las llaves del diccionario son tuple(x)
            tuple(x) consiste en un diccionario con los puntos de points más sus simétricos ordenados al rededor de x.
        
        '''
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
        
        '''
        
            Función para el cálculo de la distancia en "tipo de orden".
            Este método es O(n\log n).

            -Parametros: 
            p: punto en points
            q: punto en points
            points: muestra de puntos donde se calculará delta
            ordered_points: Diccionario que almacenará la información de ordenamiento respecto a points.

            -return:
            La distancia entre p y q. Es un entero.
        '''
        # Condición de distancia para puntos fuera de "points"

        if (q not in points) or (p not in points):

            return 0

        else:

            # Inicialización de la distancia.

            distance = 0

            # Lista que consiste en los puntos distintos a p y q.

            npnq = [point for point in points if point != p and point != q]

            for x in npnq:
                    
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
   
    def centroid(S: list) -> list:
        
        '''
            Función para calcular el centroide de un conjunto de puntos S respecto a una distancia cualquiera.
            La implementación se realiza dinámicamente aprovechando delta(p,q)=delta(q,p)

            En esta función se utiliza la función definida para este problema (delta), pero se espera que funcione para cualquier distancia.
            El centroide se calcula como aquel punto p que minimice la suma de distancias \sum delta(p,q_{i}).
            Este método es O(k^{3}\log k) donde k es el tamaño de S.

            Verificamos que S sea no vacío en caso contrario el centroide se define como el vector (0,0)

            - Parametros:
            S: Lista que contiene vectores (representados como listas)

            -return:
            Regresa un punto que es el centroide calculado.
        
        '''
        
        if S:

            op = ordered_points(points=S) # O(k\log k)

            # Inicialización de la distancia mínima 

            min_distance = float('inf')

            centroid = None

            # Diccionario para realizar programación dinámica y almacenar distancias ya calculadas.

            cache = dict()

            for i in range(len(S)):

                p = S[i]

                sum_distance = 0

                for j in range(len(S)):

                    q = S[j]

                    if (i,j) in cache:

                        distance = cache[(i,j)]
                    
                    else:

                        distance = delta(p=p, q=q, points=S, ordered_points=op)

                        cache[(i,j)] = distance

                        cache[(j,i)] = cache[(i,j)]
                
                    sum_distance += distance

                if sum_distance < min_distance:
                    
                    min_distance = sum_distance

                    centroid = p

            return centroid
        
        else:

            return [0,0]

    '''
        Algoritmo de Lloyd. 
        Devuelve un diccionario para representar los clusters y los centros.

    '''
    def plot_cluster(clusters, centers, iteration):
        plt.figure()
        colors = ['g', 'b', 'c', 'm', 'y', 'k']

        for cluster_id, points in clusters.items():
            x_coords, y_coords = zip(*points)
            x_center, y_center = centers[cluster_id]
            color = colors[cluster_id % len(colors)]
            
            plt.scatter(x_coords, y_coords, c=color, label=f'Cluster {cluster_id+1}')
            plt.scatter(x_center, y_center, c='r', s=30, linewidths=3)
        plt.title(f"Iteration {iteration}")
        plt.legend()
        plt.savefig(fr'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\combinatorial\comb_{iteration}.png')
        #plt.show()

    op = ordered_points(points=points)
    
    # Elección aleatoria de k centros

    centers = random.sample(points, k)

    # Variable indicadora para el número de iteraciones.

    f = 1

    while f < 10:

        # Diccionario para representar los clusters

        cluster = dict()

        for j in range(k): cluster[j] = list()

        # Tiempo del siguiente ciclo: O(kn^{2}\log n)
        
        for p in points:

            # Distancia del punto p a cada centro.

            distances = [delta(p=p, q=center, points=points, ordered_points=op) for center in centers]

            # Encontramos la distancia de p a un centro mínima.
            
            min_distance_index = np.argmin(distances)

            # Asignación del punto p al cluster "min_distance_index"

            cluster[min_distance_index].append(p)

        if plot:

            plot_cluster(clusters=cluster, centers=centers, iteration=f)
        
        # Cálculo de nuevos centros

        updated_centers = list()

        # Tiempo del siguiente ciclo: O(k^{4}\log k)

        for j in range(k):

            # Se añade el centroide (respecto a "delta") al nuevo conjunto de centros

            updated_centers.append(centroid(cluster[j]))
            
        f += 1
        
        if centers == updated_centers: break
        
        else: centers = updated_centers

    print(f'{f} iteraciones.')
    
    return {'cluster': cluster, 'centers': centers}

def cluster(k: int, num_points: list) -> None:

    def clustering_statistics(n: int, k: int) -> list:

        '''
            Programa para extraer estadísticas del cluster hecho por order_type_clustering

            - Parametros:
            n: Número de puntos de la gráfica a analizar.
            k: Cantidad de clusters

            Es necesario que se cuente con los archivos rectilinear_crossing_number.pkl
        '''
        # Lectura de puntos

        file=open(r'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\rectilinear_crossing_number.pkl','rb')

        D=pickle.load(file, encoding='latin1')

        points = D[n]['pts']

        clustering = order_type_clustering(points=points, k=k)

        # Guarda el diccionario en un archivo de texto como JSON
        with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(len(points))+'pts\\'+str(k)+'clusters.txt', "w") as f:
            
            json.dump(clustering, f)

        # Eliminamos contenido previo.
        open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(len(points))+'pts\\'+str(k)+'clusters_info.txt', "w").close()
        
        if os.path.exists(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(len(points))+'pts_\\'+str(k)+'clusters.txt'):
            
            os.remove(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(len(points))+'pts_\\'+str(k)+'clusters.txt')

        pts_statistics = list()

        for i in range(k):

            cluster = clustering['cluster'][i]

            try: pts = D[len(cluster)]['pts']
            except: continue

            optimal = PyDCG.crossing.count_crossings_py(pts)
            crossing = PyDCG.crossing.count_crossings_py(cluster)
            
            pts_statistics.append(round(len(cluster) / len(points),3))
            if crossing == 0: pts_statistics.append(1)
            else: pts_statistics.append(round(optimal / crossing,3))

            with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(len(points))+'pts\\'+str(k)+'clusters_info.txt', "a") as f:
                f.write(f'\nNum. Puntos Cluster {i+1}: {len(cluster)}')
                f.write(f'\nCrossing Number Cluster {i+1}: {crossing}')
                f.write(f'\nOptimal Crossing Number {i+1}: {optimal}')
                f.write(f'\nRazón Número Puntos y Cluster {i+1}: {round(len(cluster) / len(points),3)}')
                if crossing == 0: f.write(f'\nRazón Óptimo y Número Cruce {i+1}: 1\n')
                else: f.write(f'\nRazón Óptimo y Número Cruce {i+1}: {round(optimal / crossing,3)}\n')

            # Desempaquetado de las coordenadas en listas separadas
            xcoordinate, ycoordinate = zip(*cluster)

            # Crea un gráfico de dispersión
            plt.scatter(xcoordinate, ycoordinate, label=f"Cluster {i+1}")

        # Agrega una leyenda
        plt.legend()

        # Agrega una leyenda
        plt.savefig(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(len(points))+'pts\\'+str(k)+'clusters\\'+str(len(points))+'pts_'+str(k)+'clusters.png')
        
        # Muestra el gráfico
        #plt.show()
        plt.close()

        return pts_statistics

    '''
        Programa que crea un archivo csv que guarda información estadística del agrupamiento en tipo de orden
        - Parámetros:
        k: tamaño del clustering
        num_poins: lista del numero de puntos a considerar. P. ej, [50,75,100,200]
    
    '''

    try:

        df = pd.read_csv(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\\'+str(k)+'clusters_statistics.csv')
    
    except:

        index = []

        for i in range(k):
            index.append(f'cl{i+1}/n')
            index.append(f'cr{i+1}/n')

        # Crea un DataFrame de pandas
        df = pd.DataFrame({'': index})

    for n in num_points:

        statistics = clustering_statistics(n,k)
        
        df[f'{n}pts'] = statistics

    df.to_csv(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\\'+str(k)+'clusters_statistics.csv', index=False)

def cluster_analysis(n: int, k: int, shoulder_arm: int) -> None:

    def save_expantion(clustering: dict, k: int) -> None:

        '''
            Función para guardar el estiramiento de cada cluster de un clustering hecho.

            - Parametros:
            clustering: agrupamiento de un conjunto de puntos. Se recomienda usar order_type_clustering
            k: número de grupos del cluster.
        '''
        
        file=open(r'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\rectilinear_crossing_number.pkl','rb')
        D=pickle.load(file, encoding='latin1')

        n = 0    
        for j in clustering['cluster'].keys():
            n += len(clustering['cluster'][j])

        # Este programa muestra realiza una transformación lineal de un conjunto de puntos
        # que infla el conjunto de puntos.
        for j in clustering['cluster'].keys():
            cluster = clustering['cluster'][j]
            if len(cluster) <= 1: continue
            
            try: optimal = D[len(cluster)]['pts']
            except: continue
            
            bp = expand(cluster=cluster, alpha=2)

            # Desempaquetado de las coordenadas en listas separadas
            xcoordinate, ycoordinate = zip(*bp)

            # Crea un gráfico de dispersión
            plt.scatter(xcoordinate, ycoordinate)

            plt.title(f"Cantidad de Puntos: {len(cluster)}\n"+f"Crossing Number: {PyDCG.crossing.count_crossings_py(cluster)}\n"+f"Optimal Crossing Number: {PyDCG.crossing.count_crossings_py(optimal)}")

            plt.savefig(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(n)+'pts\\'+str(k)+'clusters\\'+'cluster_'+str(int(j)+1)+'.png')

            # Muestra el gráfico
            #plt.show()
        
        return None
    
    def expand(cluster: list, alpha=1) -> list:

        '''
            Función para expandir un conjunto de puntos por un parametro alpha
            
            -Parametros:
            cluster: conjunto de puntos
            alpha: factor de estiramiento
        '''

        sorted_cluster = sorted(cluster, key=lambda vector: vector[0], reverse=True)
        sup = sorted_cluster[0][0]
        # Lista para almacenar los vectores q-p
        pts = list()
        for p in sorted_cluster:
            for q in sorted_cluster[sorted_cluster.index(p)+1:]:
                pts.append([q[0]-p[0], q[1]-p[1]])
        v = [sum([x[0] for x in pts]) / len(pts), sum([x[1] for x in pts]) / len(pts)]
        w = [(-1)*v[1], v[0]]
        
        # T = transformacion lineal
        T = [v, [alpha * sup * w[0], alpha * sup * w[1]]]

        blow_up = list()

        for p in cluster:

            blow_up.append([(T[0][0]*p[0])+(T[0][1]*p[1]), (T[1][0]*p[0])+(T[1][1]*p[1])])
        
        return blow_up

    '''
        Función para realizar un clustering en tipo de orden a un cluster.

        -Parametros:
        n: cantidad de puntos del conjunto de puntos original.
        k: número de clusters.
        shoulder_arm: indica el brazo u hombro que se desea analizar nuevamente con order_type_clustering.
    '''
    
    try:
        with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(n)+'pts\\'+str(k)+'clusters.txt', 'r') as f:
            clustering = json.loads(f.read())
        f.close()
    except: raise('No existe archivo de clustering.')

    save_expantion(clustering=clustering, k=k)
    # Brazo a analizar del clustering hecho para el conjunto de n puntos en k clusters.
    arm = clustering['cluster'][str(shoulder_arm-1)]

    # otc = order_type_clustering
    otc = order_type_clustering(expand(arm), k)
    # Lectura de puntos

    file=open(r'C:\Users\wamjs\OneDrive\Documentos\Python\ruy\rectilinear_crossing_number.pkl','rb')

    D=pickle.load(file, encoding='latin1')

    for i in range(k):

        cluster = otc['cluster'][i]

        try: pts = D[len(cluster)]['pts']
        except: continue

        optimal = PyDCG.crossing.count_crossings_py(pts)
        crossing = PyDCG.crossing.count_crossings_py(cluster)

        print(f'\nNum. Puntos Cluster {i+1}: {len(cluster)}')
        print(f'\nCrossing Number Cluster {i+1}: {crossing}')
        print(f'\nOptimal Crossing Number {i+1}: {optimal}')
        print(f'\nRazón Número Puntos y Cluster {i+1}: {round(len(cluster) / len(pts),3)}')
        if crossing == 0: print(f'\nRazón Óptimo y Número Cruce {i+1}: 1\n')
        else: print(f'\nRazón Óptimo y Número Cruce {i+1}: {round(optimal / crossing,3)}\n')

        # Desempaquetado de las coordenadas en listas separadas
        xcoordinate, ycoordinate = zip(*cluster)

        # Crea un gráfico de dispersión
        plt.scatter(xcoordinate, ycoordinate, label=f"Cluster {i+1}")

    # Agrega una leyenda
    plt.legend()

    # Agrega una leyenda

    # Muestra el gráfico
    plt.show()