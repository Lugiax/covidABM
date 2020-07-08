#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from ambiente import Mundo
    from individuo import Individuo
except:
    from Ambiente.ambiente import Mundo
    from Individuos.individuo import Individuo

from modelos import Simple
import datetime
import time
import argparse
from utils import calcular_error

parser = argparse.ArgumentParser()
parser.add_argument('--evitar_agentes', type=bool)
parser.add_argument('--distancia_paso', type=int)
parser.add_argument('--prob_movimiento', type=float)
#parser.add_argument('--frac_mov_nodos', type=float)
parser.add_argument('--prob_contagiar', type=float)
parser.add_argument('--prob_infectarse', type=float)
parser.add_argument('--radio_de_infeccion', type=int)
parser.add_argument('--tamano', type=int, default=150)
parser.add_argument('--inds_x_agente', type=int, default=5)
parser.add_argument('--expuestos_iniciales', type=int, default=0)
parser.add_argument('--reduccion_mov', type=float)
parser.add_argument('-o', '--salida', type=str, default='resultado0.pk')
parser.add_argument('--n_dias', type=int, default=293)## Los días que le quedan al año a partir del dia cero
args = parser.parse_args().__dict__

attrs_individuos = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':1,
                    #'frac_mov_nodos':0.01,
                    #Ante la enfermedad
                    'prob_contagiar': 0.5,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }
modelo_params = {
                    'tamano':5,
                    'inds_x_agente':500,
                    'dia_cero':datetime.datetime(2020,3,13),
                    'expuestos_iniciales':5,
                    'reduccion_mov': None
                }


print(f'\nAjustes seleccionados:')
ind_attrs = {**attrs_individuos}
mod_param = {**modelo_params}
for arg in args:
    if args.get(arg) is not None and arg!='salida' and arg!='n_dias':
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
modelo = Simple(Mundo, Individuo,
                mod_param,
                ind_attrs)
t1 = time.time()
modelo.correr(args['n_dias'], show=True)
tf = time.time()
corrida = modelo.datacollector.get_model_vars_dataframe()
corrida.to_pickle(args['salida'])
print(f'\tResultados guardados correctamente en {args["salida"]}')
ts1 = t1-t0
tsf = (tf-t1)/60
print(f'Tiempo de configuración {ts1} segundos.\nTiempo de ejecución {tsf} minutos')

#import datetime
#dia_inicio = datetime.datetime(2020,4,17)
#dia_final = datetime.datetime(2020,4,27)
#print('\tCalculando el error...')
#error = calcular_error(corrida, 
#					   intervalo = (dia_inicio, dia_final),
#					   inds_x_agente = mod_params['inds_x_agente']
#					  ).sum(axis=1)
#print(f'El error total es {error.sum()}\n---------------------------------------')
