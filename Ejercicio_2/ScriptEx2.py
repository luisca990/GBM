def calcular_puntos(orden_llegada, sistema_puntos):
    puntos = [0] * len(orden_llegada)
    for i, posicion in enumerate(orden_llegada):
        if posicion <= len(sistema_puntos) - 1:
            puntos[i] = sistema_puntos[posicion]
    return puntos

def dterminar_campeon(puntos):
    puntos_max = max(puntos)
    campeon = [i+1 for i, p in enumerate(puntos) if p == puntos_max]
    return campeon

while True:
    ordenes_llegada = []
    sistemas_puntos = []
    campeones = []

    G, P = map(int, input().split(" "))
    if G == 0 and P == 0:
        break
    
    for g in range(G):
        orden_llegada = list(map(int, input().split(" ")))
        ordenes_llegada.append(orden_llegada)
    
    S = int(input())
    
    for s in range(S):
        sistema_puntos = list(map(int, input().split(" ")))
        sistemas_puntos.append(sistema_puntos)
    
    
    for sistema_puntos in sistemas_puntos:
        total_points = [0] * P
        for orden_llegada in ordenes_llegada:
            points = calcular_puntos(orden_llegada, sistema_puntos)
            total_points = [sum(x) for x in zip(total_points, points)]
        campeones.append(dterminar_campeon(total_points))
    
    for campeon in campeones:
        print(' '.join(map(str, campeon)))

