import subprocess

prueba_numero = 2


comando_windows = f'type PruebaUnitaria{prueba_numero}.txt | python ../ScriptEx2.py > Salidas/SalidaPrueba{prueba_numero}.txt'

comando_linux = f'python  ../ScriptEx2.py < PruebasUnitarias{prueba_numero}.txt > Salidas/SalidaPrueba{prueba_numero}.txt'

if subprocess.os.name == 'nt':
    subprocess.run(comando_windows, shell=True)
else:  
    subprocess.run(comando_linux, shell=True)