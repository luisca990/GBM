import subprocess

prueba_numero = 5


comando_windows = f'type PruebaUnitaria{prueba_numero}.txt | python ../ScriptEx3.py > Salidas/SalidaPrueba{prueba_numero}.txt'

comando_linux = f'python  ../ScriptEx3.py < PruebasUnitarias{prueba_numero}.txt > Salidas/SalidaPrueba{prueba_numero}.txt'

if subprocess.os.name == 'nt':
    subprocess.run(comando_windows, shell=True)
else:  
    subprocess.run(comando_linux, shell=True)