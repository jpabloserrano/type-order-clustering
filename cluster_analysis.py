import json
import matplotlib.pyplot as plt


def blow_up(clustering: list, alpha=1) -> None:
    k = 0    
    for j in clustering['cluster'].keys():
        k += len(clustering['cluster'][j])

    # Este programa muestra realiza una transformación lineal de un conjunto de puntos
    # que infla el conjunto de puntos.
    for j in clustering['cluster'].keys():
        cluster = clustering['cluster'][j]
        sorted_cluster = sorted(cluster, key=lambda vector: vector[0], reverse=True)
        # Lista para almacenar los vectores q-p
        pts = list()
        for p in sorted_cluster:
            for q in sorted_cluster[sorted_cluster.index(p)+1:]:
                pts.append([q[0]-p[0], q[1]-p[1]])
        v = [sum([x[0] for x in pts]) / len(pts), sum([x[1] for x in pts]) / len(pts)]
        w = [(-1)*v[1], v[0]]
        
        # T = transformacion lineal
        T = [v, [alpha * w[0], alpha * w[1]]]

        blow_up = list()

        for p in cluster:

            blow_up.append([(T[0][0]*p[0])+(T[0][1]*p[1]), (T[1][0]*p[0])+(T[1][1]*p[1])])

        # Desempaquetado de las coordenadas en listas separadas
        xcoordinate, ycoordinate = zip(*blow_up)

        # Crea un gráfico de dispersión
        plt.scatter(xcoordinate, ycoordinate)

        plt.title('Cluster ' + str(int(j)+1))

        plt.savefig(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\figs\\'+str(k)+'pts\\'+'cluster_'+str(int(j)+1)+'.png')

        # Muestra el gráfico
        plt.show()

with open(r'C:\Users\wamjs\OneDrive\Documentos\Cinvestav\type-order-clustering\clustering\1000pts_6clusters.txt', 'r') as f:
    clustering = json.loads(f.read())
f.close()

blow_up(clustering=clustering, alpha=10000)