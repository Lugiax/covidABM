#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2
from modelos import Modelo
from utils import leer_historico, calcular_error
from AG import AG

import pandas as pd
import numpy as np
import datetime
import pickle as pk
import time

def error(*args):
	dia_cero = datetime.datetime(2020,4,17)
	un_dia = datetime.timedelta(days = 1)
	dia_final = datetime.datetime(2020,5,15)#list(hist.columns.get_level_values(0))[-1]
	n_dias = int((dia_final-dia_cero)/un_dia)+1
	"""
	hist = leer_historico() 
	hist = hist.iloc[:, hist.columns.get_level_values(1)=='Activos']
	hist = hist.iloc[:, hist.columns.get_level_values(0)>=dia_cero]
	hist = hist.iloc[:, hist.columns.get_level_values(0)<=dia_final]
	"""
	#print(f'Dia cero {dia_cero}, dia final {dia_final}. Dias totales {n_dias}')
	attrs_individuos = {#De comportamiento
	                    'evitar_agentes': False if args[0]<0.5 else True,
	                    'distancia_paso': 1,
	                    'prob_movimiento': args[1],#0.5,
	                    'frac_mov_nodos': args[2], #0.01,
	                    #Ante la enfermedad
	                    'prob_contagiar': args[3], #0.2,
	                    'prob_infectarse': args[4], #0.1,
	                    'radio_de_infeccion': 0
	                    }
	modelo_params = {
	                    'area':200,
	                    'inds_x_agente':5,
	                    'dia_cero':dia_cero,
	                    'prop_inf_suscep': args[5]
	                }
	modelo = Modelo(Mundo, Individuo_2,
	                modelo_params,
	                attrs_individuos)
	modelo.correr(n_dias*4)

	simu = modelo.datacollector.get_model_vars_dataframe()

	error_cols = calcular_error(simu, (dia_cero, dia_final),
								modelo_params['inds_x_agente'])
	return error_cols.sum()
	



ag=AG(deb=True)
ag.parametros(Nind=16,Ngen=50,optim=0, pres=0.001, procesos = 16)
ag.variables(variables=[['evitar_agentes', 0, 1],
	                    ['prob_movimiento', 0.01, 0.5],#0.5,
	                    ['frac_mov_nodos', 0.001, 0.3], #0.01,
	                    ['prob_contagiar', 0.001, 0.4], #0.2,
	                    ['prob_infectarse', 0.001, 0.4], #0.1,
	                    ['prop_inf_suscep', 0, 4]
	                    ])


ag.Fobj(error)
t1=time.time()
res=ag.start()
print('Tiempo de cómputo {}ms'.format(int(time.time()-t1)))

with open('resultados/resAG1.pk', 'wb') as f:
	pk.dump(res, f)
print('Los resultados son:')

for val, tipo in zip(res[0],['evitar_agentes','prob_movimiento',
						'frac_mov_nodos','prob_contagiar','prob_infectarse','prop_inf_suscep']):
	print(f'{tipo}: {val}')
print(f'Valor más pequeño de la función objetivo: {res[1]}')

