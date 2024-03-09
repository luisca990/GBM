def es_palindromo(palabra):
    palabra = palabra.replace(" ", "").lower()
    palabraInvertida = palabra[::-1]
    return palabra == palabraInvertida


def imprimir_resultado(palabra):
    if es_palindromo(palabra):
        print(f"{palabra} es un palíndromo")
    else:
        print(f"{palabra} no es un palíndromo")

    
if __name__ == "__main__":
    palabraIngresada = input("Ingrese la palabra o frase a validar\n")
    imprimir_resultado(palabraIngresada)
