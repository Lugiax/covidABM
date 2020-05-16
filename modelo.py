#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:50:13 2020

@author: carlos
"""

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2
from modelos import Modelo
attrs_individuos = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.5,
                    'frac_mov_nodos':0.01,
                    #Ante la enfermedad
                    'prob_contagiar': 0.2,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }
modelo_params = {
                    'area':175,
                    'inds_x_agente':5
                }

numero_de_pasos = 1000

ajustes = [#{},
    #{'evitar_agentes':True},
    #{'prob_movimiento':0.3},
    #{'prob_movimiento':0.1},
    #{'frac_mov_nodos':0.1},
    #{'frac_mov_nodos':0.001},
    #{'evitar_agentes':True, 'prob_movimiento':0.3},
    {'evitar_agentes':True, 'prob_movimiento':0.3, 'frac_mov_nodos':0.1},
    {'evitar_agentes':True, 'prob_movimiento':0.1, 'frac_mov_nodos':0.001},
    {'evitar_agentes':True, 'prob_movimiento':0.1, 'frac_mov_nodos':0.001, 'radio_de_infeccion':2}
    ]


for i, ajuste in enumerate(ajustes):
    print(f'\nEjetutando ajuste {i} \nAjustes:')
    atributos = {**attrs_individuos}
    for tipo in ajuste:
        print(f'\t- {tipo} = {ajuste[tipo]}')
        atributos[tipo] = ajuste[tipo]

    modelo = Modelo(Mundo, Individuo_2,
                    modelo_params,
                    atributos)
    modelo.correr(numero_de_pasos)
    data = modelo.datacollector.get_model_vars_dataframe()
    data.to_pickle(f'resultados/ajuste4_{i}.pk')
    print('\tResultados guardados correctamente\n---------------------------------------')

    del(modelo)
    del(data)
