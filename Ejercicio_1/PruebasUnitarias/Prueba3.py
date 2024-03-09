import sys
sys.path.append('../')
from ScriptEx1 import  * 

ruta_archivo_salida ="Salidas/SalidaPrueba3.txt"
def prueba_palindromo():
    palabra = "La ruta natural"
    resultado_esperado = True
    resultado_obtenido = es_palindromo(palabra)
    return f"Prueba unitaria para \"{palabra}\" EXITOSA" if resultado_esperado == resultado_obtenido else f"Prueba unitaria para \"{palabra}\" FALLIDA"

def escribir_archivo_prueba():
    try:
        with open(ruta_archivo_salida, 'w') as f:
            resultado = prueba_palindromo()
            f.write(resultado)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if escribir_archivo_prueba():
        print(f"Prueba unitaria ejecutada, revisar el archivo {ruta_archivo_salida}")