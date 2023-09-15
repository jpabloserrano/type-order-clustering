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

    o1 = orientation(P, Q, X)
    
    o2 = orientation(P, Q, Y)
    
    o3 = orientation(X, Y, P)
    
    o4 = orientation(X, Y, Q)

    # Los puntos son colineales o los segmentos se cruzan
    
    return o1 != o2 and o3 != o4

#Función para contar el par de puntos que cruzan a P y Q.
def count_crossing_pairs(P: list, Q: list, points: list) -> int:
    
    count = 0

    # Iterar sobre cada par de puntos
    
    for i in range(len(points)):
    
        for j in range(i + 1, len(points)):
    
            if is_crossing(P, Q, points[i], points[j]):
    
                count += 1

    return count

# Ejemplo de uso

p = [0, 0]

q = [1, 2]

points = [[1, 1], [2, 7], [3, 5], [0, 3], [4, 0], p, q]

result = count_crossing_pairs(p, q, points)

print("Número de pares de puntos con líneas que cruzan el segmento de p a q:", result)
