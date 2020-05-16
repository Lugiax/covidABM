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
                    'evitar_sintomaticos': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.5,
                    'frac_mov_nodos':0.1,
                    'activar_cuarentena': False,
                    'quedate_en_casa': False,
                    #Ante la enfermedad
                    'prob_contagiar': 0.2,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }

modelo_params = {
                    'area':75,
                    'inds_x_agente':500
                }

modelo = Modelo(Mundo, Individuo_2,
                modelo_params,
                attrs_individuos)

#modelo.mundo.ver_espacio('Mérida')
modelo.correr(100)
data = modelo.datacollector.get_model_vars_dataframe()
data.to_pickle('Pruebas/prueba1.pk')
data[['Suceptibles', 'Expuestos', 'Infectados', 'Recuperados']].plot()