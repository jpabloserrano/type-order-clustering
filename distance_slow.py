# Este programa calcula la distancia en tipo de orden de un par de puntos p y q.

# Función para conocer la orientacióno de 3 puntos A, B, C
def orientation(A: list, B: list, C: list) -> str:
    
    # Función para calcular la orientación entre tres puntos
    
    det = ((B[0]-A[0])*C[1])+((A[0]-C[0])*B[1])+((C[0]-B[0])*A[1])
    
    if det == 0:
    
        return 'col'  # Colineales
    
    return 'izq' if det > 0 else 'der'  # En sentido horario o antihorario

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

# Ejemplo de uso

p = [3,3]

q = [0, 2]

points = [[0, 3], [4, 0], p, q]

print(count_crossing_pairs(p,q,points))