def saltos_min(x):
    salto = 0
    posicion = 0
    while posicion < x:
        salto += 1
        posicion += salto
        if posicion == x:
            return salto
        elif posicion > x:
            return salto if (posicion - x) % 2 == 0 else salto + 1


total_saltos = []
total_x = []
x = int(input())

for i in range(x):
    x = int(input())
    total_x.append(x)

for j in range(len(total_x)):
    total_saltos.append(saltos_min(total_x[j]))

for p in total_saltos:
    print(p)
