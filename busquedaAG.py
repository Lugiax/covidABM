#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2
from modelos import Modelo
from utils import leer_historico, convertir_corrida
from AG import AG

import pandas as pd
import numpy as np
import datetime

def error():
	dia_cero = datetime.datetime(2020,4,20)
	un_dia = datetime.timedelta(days = 1)
	dia_final = datetime.datetime(2020,5,1)#list(hist.columns.get_level_values(0))[-1]
	n_dias = int((dia_final-dia_cero)/un_dia)

	hist = leer_historico() 
	hist = hist.iloc[:, hist.columns.get_level_values(1)=='Activos']
	hist = hist.iloc[:, hist.columns.get_level_values(0)>=dia_cero]
	hist = hist.iloc[:, hist.columns.get_level_values(0)<=dia_final]


	print(f'Dia cero {dia_cero}, dia final {dia_final}. Dias totales {n_dias}')	
	
	attrs_individuos = {#De comportamiento
	                    'evitar_agentes': False,
	                    'distancia_paso': 1,
	                    'prob_movimiento':0.5,
	                    'frac_mov_nodos':0.01,
	                    #Ante la enfermedad
	                    'prob_contagiar': 0.2,
	                    'prob_infectarse': 0.1,
	                    'radio_de_infeccion': 0
	                    }
	modelo_params = {
	                    'area':5,
	                    'inds_x_agente':500
	                }
	modelo = Modelo(Mundo, Individuo_2,
	                modelo_params,
	                attrs_individuos)
	modelo.correr((n_dias+1)*4)
	simu = modelo.datacollector.get_model_vars_dataframe()
	simu = convertir_corrida(simu)
	
	#simu = leer_corrida('resultados/ajusteprueba_2.pk')
	simu = simu.iloc[:,simu.columns.get_level_values(1)=='I']
	simu = pd.DataFrame(simu.iloc[:,1:].values.T, 
		index = simu.iloc[:,1:].columns.get_level_values(0),
		columns = list(map(lambda x: dia_cero+un_dia*x, list(simu.index))))
	simu = simu.loc[simu.index.isin(list(hist.index))]

	#list(zip(simu.index, hist.index))#Para asegurar coincidencia de municipios
	#print(simu)
	#print(hist)
	muns = list(simu.index)
	res = np.zeros(simu.shape)
	for i, mun in enumerate(muns):
	    res[i] = (hist.loc[mun].values-simu.loc[mun].values)**2
	#suma = res.sum(axis=0)
	return res.sum()

print(error())