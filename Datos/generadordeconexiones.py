import pickle as pk
import os
import numpy as np

#Se recupera la 
with open('datos.pk', 'rb') as f:
    datos = pk.load(f)


municipios = list(datos.keys())
municipios.remove('Yucatán')
municipios = sorted(municipios, key = lambda x: datos[x]['num'])
print(municipios)
n_tot = len(municipios)
print(f'Se tienen en total {n_tot} municipios')
n_mun = 0
n_reg = 0
if os.path.exists('conexiones.csv'):
    with open('conexiones.csv','r') as f:
        lineas = f.readlines()
        n_reg = len(lineas)

with open('conexiones.csv', 'a+') as f:
    if n_reg>0:
        print(f'El archivo cuenta con {n_reg} entradas.'
              f'Se continuará a partir de {municipios[n_mun+n_reg]}')
        n_mun = n_mun+n_reg
    else:
        print('Se comenzará a generar las conexiones por municipio')
    while n_mun<n_tot:
        mun_actual = municipios[n_mun]
        print(f'\nConexiones con el municipio {mun_actual}')
        terminado = False
        pesos = ['0' for i in range(len(municipios))]
        agregados = []
        while not terminado:
            seleccion = input('\tNombre del municipio a conectar > ')
            if seleccion.lower() == 'c':
                break
            elif seleccion.lower() == 'terminar':
                n_mun = n_tot+1
                terminado = True
                break
            elif seleccion in agregados:
                print('\t --- Ya agregado, seleccione otro')
                continue
            while seleccion not in municipios:
                posibles = ', '.join([m for m in municipios if m.startswith(seleccion)])
                print(f'\t --- Coincidencias: {posibles}')
                seleccion = input('\t --- Seleccione nuevamente > ')
                if seleccion in agregados:
                    print('\t --- Ya agregado, seleccione otro')
            indice_mun = municipios.index(seleccion)
            agregados.append(seleccion)
            peso = input('\tIngrese el peso de la conexión (0-1) > ')
            pesos[indice_mun] = peso
        f.write(mun_actual+','+','.join(pesos)+'\n')
        n_mun+=1
print('Proceso terminado----------')
                
            
        
    
    
    
 