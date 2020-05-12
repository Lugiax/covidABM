#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle as pk
import os
import numpy as np
from collections import OrderedDict


#Se recupera la 
with open('datos.pk', 'rb') as f:
    datos = pk.load(f)


municipios = list(datos.keys())
#municipios = sorted(municipios, key = lambda x: datos[x]['num'])
#print(municipios)
n_tot = len(municipios)
print(f'Se tienen en total {n_tot} municipios')


try:
    with open('conexiones4.pk', 'rb') as l:
        L = pk.load(l)
        print('Se abre el archivo guardado')
except:
    L = OrderedDict()

inicio = True
while inicio:
    mun_actual = input('Municipio central > ')
    while mun_actual not in municipios:
        mun_actual = input('---Selecciona uno válido > ')
    #indice_mun_1 = municipios.index(mun_actual)
    if L.get(mun_actual, None) is not None:
        cargados = [mun for mun,p in L.get(mun_actual)]
        print(f'Para este municipio se tienen datos cargados de '
            f'{", ".join(cargados)}')
    else:
        L[mun_actual] = set()

    print(f'\nConexiones con el municipio {mun_actual}')
    terminado = False
    #---agregados = []
    while not terminado:
        guardar = True
        seleccion = input('\tNombre del municipio a conectar > ')
        if seleccion.lower() == 'c':
            break
        elif seleccion.lower() == 'terminar':
            #---n_mun = n_tot+1
            terminado = True
            inicio = False
            break
        #---elif seleccion in agregados:
        #---    print('\t --- Ya agregado, seleccione otro')
        #---    continue
        while seleccion not in municipios:
            posibles = ', '.join([m for m in municipios if m.startswith(seleccion)])
            print(f'\t --- Coincidencias: {posibles}')
            seleccion = input('\t --- Seleccione nuevamente > ')
            #---if seleccion in agregados:
            #---    print('\t --- Ya agregado, seleccione otro')
        #---indice_mun_2 = municipios.index(seleccion)
        #---agregados.append(seleccion)
        while True:

            peso = input('\t  -Ingrese los pesos de las conexiones (0-1),(0-1) > ')
            try:
                pesos = [float(x) for x in peso.split(',')]
            except:
                print('\t---***Ingresar valores válidos***---')
                continue
            if len(pesos)==1 and pesos[0]==0.0:
                print(f'\t  **Se va a eliminar {seleccion}...', end='')
                for _x, eliminar_a in enumerate(L[mun_actual]):
                    if eliminar_a[0]==seleccion:
                        print(f'Eliminado {L[mun_actual].pop(_x)}**')
                        for _x2, eliminar_a2 in enumerate(L[seleccion]):
                            if eliminar_a2[0]==mun_actual:
                                print(f'Eliminado {L[seleccion].pop(_x2)}**')
                guardar = False

            if len(pesos)==1:
                pesos += pesos
            break
        
        if guardar:
            L[mun_actual].append((seleccion, pesos[0]))
            L[seleccion].append((mun_actual, pesos[1]))
        #---matriz[indice_mun_1, indice_mun_2] = pesos[0]
        #---matriz[indice_mun_2, indice_mun_1] = pesos[1]

with open('conexiones4.pk', 'wb') as p:
    pk.dump(L, p)

print('Proceso terminado----------')