from PyDCG import geometricbasics

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
# Recibe como parámetro "ordered_points" que debe contener un diccionario de las listas ordenadas por punto x.
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
                
                # Puntos simétricos de p y q respecto a x.
                
                sym_p = [(2 * x[0]) - p[0], (2 * x[1]) - p[1]]

                sym_q = [(2 * x[0]) - q[0], (2 * x[1]) - q[1]]

                # spax = sorted points by angle from x
                
                spax = ordered_points[tuple(x)]['ordered_points']

                sym_spax = ordered_points[tuple(x)]['symmetric_points']

                # Almacenamos la posicion del orden de p y q así como sus simétricos.

                index_p, index_q = spax.index(p), spax.index(q) # Esta línea se puede hacer en O(log n)
                
                sym_index_p, sym_index_q = sym_spax.index(sym_p), sym_spax.index(sym_q) # Esta línea se puede hacer en O(log n)

                spin = orientation(x, p, q)

                # Verificamos orientación

                if spin == 'izq':
                    
                    if index_p < index_q:
                    
                        distance += len(spax[index_p+1:index_q]) + len(spax[sym_index_q+1:sym_index_p])

                    elif index_p > index_q:

                        distance += len(spax[index_p+1:]) + len(spax[1:index_q]) + len(spax[sym_index_q+1:]) + len(spax[1:sym_index_p])

                elif spin == 'der':

                    if index_p > index_q:
                    
                        distance += len(spax[index_q+1:index_p]) + len(spax[sym_index_p+1:sym_index_q])

                    elif index_p < index_q:

                        distance += len(spax[index_q+1:]) + len(spax[1:index_p]) + len(spax[sym_index_p+1:]) + len(spax[1:sym_index_q])

        return int(distance / 2)
    
# Ejemplo de uso

p = [4,4]

q = [-4,0]

points = [[0, 3], [5, 0], [2,4], [2,2], [0,-4], p, q]

result = delta(p, q, points, ordered_points(points))

print(f"Distancia entre {p} y {q}: {result}")
