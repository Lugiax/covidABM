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
if not os.path.exists('conexiones.csv'):
    with open('conexiones.csv','w') as f:
        pass


if os.path.exists('conexiones.pk'):
    with open('conexiones.pk', 'rb') as m:
        matriz = pk.load(m)
else:
    matriz = np.zeros((n_tot, n_tot))

inicio = True
while inicio:
    mun_actual = input('Municipio central > ')
    while mun_actual not in municipios:
        mun_actual = input('Municipio central > ')
    indice_mun_1 = municipios.index(mun_actual)
    if np.any(matriz[indice_mun_1]):
        cargados = [municipios[i] for i, val in enumerate(matriz[indice_mun_1])\
                    if val>0]
        print(f'Para este municipio se tienen datos cargados de {", ".join(cargados)}')

    print(f'\nConexiones con el municipio {mun_actual}')
    terminado = False
    agregados = []
    while not terminado:
        seleccion = input('\tNombre del municipio a conectar > ')
        if seleccion.lower() == 'c':
            break
        elif seleccion.lower() == 'terminar':
            n_mun = n_tot+1
            terminado = True
            inicio = False
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
        indice_mun_2 = municipios.index(seleccion)
        agregados.append(seleccion)
        peso = input('\tIngrese los pesos de las conexiones (0-1),(0-1) > ')
        pesos = [float(x) for x in peso.split(',')]
        matriz[indice_mun_1, indice_mun_2] = pesos[0]
        matriz[indice_mun_2, indice_mun_1] = pesos[1]

with open('conexiones.pk', 'wb') as p:
    pk.dump(matriz, p)
np.savetxt('conexiones.csv', matriz, delimiter=',',
            fmt='%1.1f')
print(matriz)
print('Proceso terminado----------')
                
            
        
    
    
    
 