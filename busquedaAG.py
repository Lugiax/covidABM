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
	dia_cero = datetime.datetime(2020,3,10)
	un_dia = datetime.timedelta(days = 1)
	dia_final = datetime.datetime(2020,4,30)#list(hist.columns.get_level_values(0))[-1]
	n_dias = int((dia_final-dia_cero)/un_dia)+1

	#print(f'Dia cero {dia_cero}, dia final {dia_final}. Dias totales {n_dias}')
	attrs_individuos = {#De comportamiento
	                    'evitar_agentes': False if args[0]<0.5 else True,
	                    'distancia_paso': 1,
	                    'prob_movimiento': args[1],#0.5,
	                    'frac_mov_nodos': args[2], #0.01,
	                    #Ante la enfermedad
	                    'prob_contagiar': args[3], #0.2,
	                    'prob_infectarse': args[4], #0.1,
	                    'radio_de_infeccion': 1
	                    }
	modelo_params = {
	                    'area':200,
	                    'inds_x_agente':5,
	                    'dia_cero':dia_cero,
	                    'expuestos_iniciales': int(args[5])
	                }
	modelo = Modelo(Mundo, Individuo_2,
	                modelo_params,
	                attrs_individuos)
	modelo.correr(n_dias*4, show=False)

	dia_inicio_datos = datetime.datetime(2020,4,17)
	simu = modelo.datacollector.get_model_vars_dataframe()

	error_cols = calcular_error(simu, (dia_inicio_datos, dia_final),
								modelo_params['inds_x_agente'])
	return error_cols.sum()
	



ag=AG(deb=False)
ag.parametros(Nind=10,Ngen=20,optim=0, pres=0.01, procesos = 4)
ag.variables(variables=[['evitar_agentes', 0, 1],
	                    ['prob_movimiento', 0.01, 1],#0.5,
	                    ['frac_mov_nodos', 0.01, 1], #0.01,
	                    ['prob_contagiar', 0.1, 1], #0.2,
	                    ['prob_infectarse', 0.1, 1], #0.1,
	                    ['expuestos_iniciales', 0, 10]
	                    ])


ag.Fobj(error)
t1=time.time()
res=ag.start()
print('Tiempo de cómputo {}ms'.format(int(time.time()-t1)))

with open('Pruebas/parametrosAG1.pk', 'wb') as f:
	pk.dump(res, f)
print('Los resultados son:')

for val, tipo in zip(res[0],['evitar_agentes','prob_movimiento',
						'frac_mov_nodos','prob_contagiar','prob_infectarse','expuestos_iniciales']):
	print(f'{tipo}: {val}')
print(f'Valor más pequeño de la función objetivo: {res[1]}')

