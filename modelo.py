#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2
from modelos import Modelo
import datetime
import argparse
from utils import calcular_error

parser = argparse.ArgumentParser()
parser.add_argument('--evitar_agentes', type=bool)
parser.add_argument('--distancia_paso', type=int)
parser.add_argument('--prob_movimiento', type=float)
parser.add_argument('--frac_mov_nodos', type=float)
parser.add_argument('--prob_contagiar', type=float)
parser.add_argument('--prob_infectarse', type=float)
parser.add_argument('--radio_de_infeccion', type=int)
parser.add_argument('--factor_area', type=int)
parser.add_argument('--inds_x_agente', type=int)
parser.add_argument('--expuestos_iniciales', type=int)
parser.add_argument('--reduccion_mov', type=float)
parser.add_argument('--dp_recuperar', type=int)
parser.add_argument('--dp_infectar', type=int)
parser.add_argument('-o', '--salida', type=str, default='resultados/resultado.pk')
parser.add_argument('--n_dias', type=int, default=500)
args = parser.parse_args().__dict__

attrs_individuos = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.8,
                    'frac_mov_nodos':0.1,
                    #Ante la enfermedad
                    'prob_contagiar': 0.5,
                    'prob_infectarse': 0.5,
                    'radio_de_infeccion': 1,
                    'dp_infectar':10,
                    'dp_recuperar':10
                    }
modelo_params = {
                    'factor_area':.1,
                    'inds_x_agente':1000,
                    'dia_cero':datetime.datetime(2020,3,10),
                    'expuestos_iniciales':1,
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
modelo = Modelo(Mundo, Individuo_2,
                mod_param,
                ind_attrs)

modelo.correr(args['n_dias']*4, show=True)
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
