def palindroma(palabra):
    palabra=palabra.replace(" ", "").lower()
    palabraInvertida = palabra[::-1]
    print("La palabra es palindromo" if palabra == palabraInvertida
            else "La palabra no es palindromo")


tipoAccion = int(input("Ingrese el número 1 para validar una palabra y el número 2 para correr las pruebras unitarias\n"))
if (tipoAccion == 1): #Agregar una nueva palabra a la lista de Palabras
    palabraIngresada = input("Ingrese la palabra a validar\n")
    palindroma(palabraIngresada)
else:
    with open('PruebasUnitarias.txt', 'r') as f:
        lineas = f.readlines()
        for i in range(len(lineas)):
            linea = lineas[i].strip()
            palabra = linea[0]
            palindroma(palabra)
