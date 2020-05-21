#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2
from modelos import Modelo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--evitar_agentes', type=bool)
parser.add_argument('--distancia_paso', type=int)
parser.add_argument('--prob_movimiento', type=float)
parser.add_argument('--frac_mov_nodos', type=float)
parser.add_argument('--prob_contagiar', type=float)
parser.add_argument('--prob_infectarse', type=float)
parser.add_argument('--radio_de_infeccion', type=int)
parser.add_argument('-o', '--salida', type=str, default='resultado0.pk')
parser.add_argument('--n_pasos', type=int, default=500)
args = parser.parse_args().__dict__

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

print(f'\nAjustes seleccionados:')
atributos = {**attrs_individuos}
for arg in args:
    if args.get(arg) is not None and arg!='salida' and arg!='n_pasos':
        atributos[arg] = args[arg]
        print(f'\t--{arg} = {args[arg]}')

        
modelo = Modelo(Mundo, Individuo_2,
                modelo_params,
                atributos)
modelo.correr(args['n_pasos'])
data = modelo.datacollector.get_model_vars_dataframe()
data.to_pickle(args['salida'])
print(f'\tResultados guardados correctamente en {args["salida"]}\n---------------------------------------')
