#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from ambiente import Mundo
    from individuo import Individuo_2
except:
    from Ambiente.ambiente import Mundo
    from Individuos.individuo import Individuo_2

from modelos import Modelo, ModeloSanaDistancia, ModeloCuarentenaEstricta, ModeloAdaptable
import datetime
import time
import argparse
from utils import calcular_error

modelos_disponibles = {'Normal': Modelo,
                       'SanaDistancia': ModeloSanaDistancia,
                       'CuarentenaEstricta': ModeloCuarentenaEstricta,
                       'Adaptable': ModeloAdaptable}

parser = argparse.ArgumentParser()
parser.add_argument('--evitar_agentes', type=bool)
parser.add_argument('--tomar_movilidad', type=bool, default=True)
parser.add_argument('--distancia_paso', type=int)
parser.add_argument('--prob_movimiento', type=float)
parser.add_argument('--frac_mov_nodos', type=float)
parser.add_argument('--prob_contagiar', type=float)
parser.add_argument('--prob_infectarse', type=float)
parser.add_argument('--radio_de_infeccion', type=int)
parser.add_argument('--factor_area', type=float)
parser.add_argument('--inds_x_agente', type=int)
parser.add_argument('--expuestos_iniciales', type=int)
parser.add_argument('--reduccion_mov', type=float)
parser.add_argument('--dp_recuperar', type=int)
parser.add_argument('--dp_infectar', type=int)
parser.add_argument('--umbral_rest', type=float)
parser.add_argument('--umbral_lib', type=float)
parser.add_argument('--liberar_sana_distancia', type=bool)
parser.add_argument('--alpha', type=float)
parser.add_argument('--tipo_modelo', type=str, default='Normal')
parser.add_argument('-o', '--salida', type=str, default='resultados/resultado.pk')
parser.add_argument('--n_dias', type=int, default=500)
args = parser.parse_args().__dict__

attrs_individuos = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.5,
                    'frac_mov_nodos':0.1,
                    #Ante la enfermedad
                    'prob_contagiar': 0.5,
                    'prob_infectarse': 0.5,
                    'radio_de_infeccion': 1,
                    'dp_infectar':10,
                    'dp_recuperar':10,
                    'alpha':0.1
                    }
modelo_params = {
                    'factor_area':.1,
                    'inds_x_agente':1000,
                    'dia_cero':datetime.datetime(2020,3,10),
                    'expuestos_iniciales':1,
                    'reduccion_mov': None,
                    'umbral_rest':0.05,
                    'umbral_lib':0.05,
                    'liberar_sana_distancia':False
                }


print(f'\nAjustes seleccionados:')
ind_attrs = {**attrs_individuos}
mod_param = {**modelo_params}
for arg in args:
    if args.get(arg) is not None and arg not in ['salida', 'n_dias', 'tipo_modelo']:
        if arg in attrs_individuos.keys():
            ind_attrs[arg] = args[arg]
            print(f'\tIndividuos: --{arg} = {args[arg]}')
        else:
            mod_param[arg] = args[arg]
            print(f'\tModelo: --{arg} = {args[arg]}')

print('\n\tLos parámetros finales del modelo son:')
tot_param = {**ind_attrs, **mod_param}
for k in tot_param:
    print(f'\t\t{k}: {tot_param[k]}')

print('Creando modelo...')
t0 = time.time()

modelo = modelos_disponibles[args['tipo_modelo']](Mundo, Individuo_2,
                                                    mod_param,
                                                    ind_attrs)
t1 = time.time()
print(f'Tiempo de configuración {t1-t0} segundos.')

modelo.correr(args['n_dias'], show=True)
tf = time.time()
print(f'Tiempo de ejecución {(tf-t1)/60} minutos')

corrida = modelo.datacollector.get_model_vars_dataframe()

#import datetime
#dia_inicio = datetime.datetime(2020,4,17)
#dia_final = datetime.datetime(2020,4,27)
#print('\tCalculando el error...')
#error = calcular_error(corrida, 
#					   intervalo = (dia_inicio, dia_final),
#					   inds_x_agente = mod_params['inds_x_agente']
#					  ).sum(axis=1)
#print(f'El error total es {error.sum()}')
corrida.to_pickle(args['salida'])
print(f'\tResultados guardados correctamente en {args["salida"]}\n---------------------------------------')
