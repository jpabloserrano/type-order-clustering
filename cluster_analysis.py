import json
import pickle
import matplotlib.pyplot as plt
import PyDCG


def blow_up(clustering: list, k: int, alpha=1) -> None:
    
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

        # Desempaquetado de las coordenadas en listas separadas
        xcoordinate, ycoordinate = zip(*blow_up)

        # Crea un gráfico de dispersión
        plt.scatter(xcoordinate, ycoordinate)

        plt.title(f"Cantidad de Puntos: {len(cluster)}\n"+f"Crossing Number: {PyDCG.crossing.count_crossings_py(cluster)}\n"+f"Optimal Crossing Number: {PyDCG.crossing.count_crossings_py(optimal)}")

        plt.savefig(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(n)+'pts\\'+str(k)+'clusters\\'+'cluster_'+str(int(j)+1)+'.png')

        # Muestra el gráfico
        plt.show()

k=4

n = 500

with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\\'+str(n)+'pts\\'+str(k)+'clusters.txt', 'r') as f:
    clustering = json.loads(f.read())
f.close()

blow_up(clustering=clustering, k=k, alpha=5)