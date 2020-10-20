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
from utils import calcular_error
import os
import pickle as pk
from utils import promediar


class CasoBase(Simple):
    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__(world_object, agent_object, params, ind_attrs, rand_seed=None)
        self.umbral_para_restriccion = params['umbral_rest']
        self.restriccion_aplicada = False
        self.dias_de_rest = [None, None]
        print(f'Agentes totales {sum(self.conteo_instantaneo)}')

    def step(self):
        self.dia = self.n_paso//self.pp_dia #es el momento del dia
        self.fecha = self.params['dia_cero']+ self.un_dia*self.dia
        if self.dia == 4:
            ##En el cuarto día, que corresponde al primer caso en
            #Yucatán, se planta un infectado. Esto para asegurar que
            #siempre habrá un infectado
            agentes = self.mundo.obtener_agentes_en_nodo('espacio')
            agentes[0].salud = self.INFECTADO

        self.conteo()
        proporcion_infectados = self.conteo_instantaneo[2]/sum(self.conteo_instantaneo)
        #print(f'#infectados: {self.conteo_instantaneo}. Proporcion de infectados {proporcion_infectados}')
        self.datacollector.collect(self)
        self.schedule.step()
        self.n_paso += 1
        self.proporcion_infectados = self.conteo_instantaneo[2]/sum(self.conteo_instantaneo)
        self.revisar_restricciones()

    def revisar_restricciones(self):
        ##En el modelo base no hay nada que revisar
        pass


    def correr(self, n_dias, show=False):
        n_steps = n_dias*self.pp_dia
        bloques = int(n_steps*0.1)
        if show:print('---- Corriendo simulación ----')
        t0 = time.time()
        for i in range(n_steps):
            self.step()
            if show and int(i%bloques) == 0:
                t1 = time.time()
                print(f'{int(i/n_steps*100)}%... {t1-t0}seg', end = '\n')
        tf = time.time()
        if show: print(f'100% en {(tf-t0)/60:.2} minutos')

    def guardar(self, path):
        carpeta = os.path.dirname(path)
        if not os.path.exists(carpeta):
            os.makedirs(carpeta)
        datos = self.datacollector.get_model_vars_dataframe()
        with open(path, 'wb') as f:
            pk.dump(datos, f)
            pk.dump(self.dias_de_rest, f)



class Cuarentena(CasoBase):

    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__(world_object, agent_object, params, ind_attrs, rand_seed=None)
        self.mundo.crear_nodo('cuarentena', 'nodo_seguridad', tamano = 1)

    def revisar_restricciones(self):
        ##Se revisa la condición para aplicar la restricción
        if not self.restriccion_aplicada and\
        self.proporcion_infectados>self.umbral_para_restriccion:
            print(f'En el día {self.dia} se aplican las restricciones')
            for a in self.schedule.agents:
                a.activar_cuarentena = True
            self.restriccion_aplicada = True

class SanaDistancia(CasoBase):

    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__(world_object, agent_object, params, ind_attrs, rand_seed=None)


    def revisar_restricciones(self):
        ##Se revisa la condición para aplicar la restricción
        if not self.restriccion_aplicada and\
        self.proporcion_infectados>self.umbral_para_restriccion:
            print(f'En el día {self.dia} se aplican las restricciones')
            for a in self.schedule.agents:
                a.evitar_agentes = True
                a.alpha = 0.5
            self.restriccion_aplicada = True

class Adaptable(CasoBase):

    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__(world_object, agent_object, params, ind_attrs, rand_seed=None)
        self.umbral_restriccion = params['umbral_rest']
        self.umbral_liberacion = params['umbral_lib']
        self.liberar_sana_distancia = params['liberar_sana_distancia']
        self.restriccion_aplicada = False
        self.aplicar_restriccion = True

    def revisar_restricciones(self):
        ##Se revisa la condición para aplicar la restricción
        if not self.restriccion_aplicada and\
        self.proporcion_infectados>self.umbral_restriccion and\
        self.aplicar_restriccion:
            fecha = self.dia_cero+self.dia*self.un_dia
            print(f'En el día {fecha} se aplican las restricciones')
            self.dias_de_rest[0] = self.dia
            for a in self.schedule.agents:
                a.evitar_agentes = True
                a.alpha = 0.5
            self.restriccion_aplicada = True
            self.aplicar_restriccion = False

        elif self.restriccion_aplicada and\
        self.proporcion_infectados<self.umbral_liberacion:
            fecha = self.dia_cero+self.dia*self.un_dia
            print(f'En el día {fecha} se levantan las restricciones')
            self.dias_de_rest[1] = self.dia
            for a in self.schedule.agents:
                a.evitar_agentes = not self.liberar_sana_distancia
                a.alpha = 0
            self.restriccion_aplicada = False


def modificar_parametros(param_ind, param_mod, params={}):
    ind_attrs = {**param_ind}
    mod_param = {**param_mod}
    for par in params:
        if params.get(par) is not None and par not in ['salida', 'n_dias', 'tipo_modelo']:
            if par in ind_attrs.keys():
                ind_attrs[par] = params[par]
                print(f'\tIndividuos: --{par} = {params[par]}')
            else:
                mod_param[par] = params[par]
                print(f'\tModelo: --{par} = {params[par]}')
    return ind_attrs, mod_param


##------------------------------------------------------------------------


##Parámetros base
ind_params = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.5,
                    #'frac_mov_nodos':0.01,
                    #Ante la enfermedad
                    'prob_contagiar': 0.5,
                    'prob_infectarse': 0.5,
                    'radio_de_infeccion': 1,
                    'dp_infectar':10,
                    'dp_recuperar':10,
                    'alpha':0.1
                    }
mod_params = {
                    'tamano':20,
                    'inds_x_agente':500,
                    'dia_cero':datetime.datetime(2020,3,13),
                    'expuestos_iniciales':1,
                    'reduccion_mov': None,
                    'umbral_rest':0.05,
                    'umbral_lib':0.05,
                    'liberar_sana_distancia':False  
                }

dias_de_simulacion = 50
"""
##Caso base
print('Base')
carpeta_de_salida = 'resultadosCasos/Base/'
modelo = CasoBase(Mundo, Individuo, mod_params, ind_params)
modelo.correr(dias_de_simulacion, show=True)
modelo.guardar(carpeta_de_salida+'base_simple1.pk')


print(promediar(['resultadosCasos/Base/base_simple1.pk'],
                rows=1, cols=2))
"""
"""
##Caso Sana Distancia
print('Sana Distancia')
carpeta_de_salida = 'resultadosCasos/DistanciamientoSocial/'
nuevos_params = {'umbral_rest': 0.1}
ind_params, mod_params = modificar_parametros(ind_params, mod_params,
                                                nuevos_params)
modelo = SanaDistancia(Mundo, Individuo, mod_params, ind_params)
modelo.correr(dias_de_simulacion, show=True)
modelo.guardar(carpeta_de_salida+'dist_soc_simple1.pk')
"""
print(promediar(['resultadosCasos/DistanciamientoSocial/dist_soc_simple1.pk'],
                rows=1, cols=2, con_fechas=True))
"""
##Caso Cuarentena
print('Cuarentena')
carpeta_de_salida = 'resultadosCasos/CuarentenaEstricta/'
nuevos_params = {'umbral_rest': 0.1}
ind_params, mod_params = modificar_parametros(ind_params, mod_params,
                                                nuevos_params)
modelo = Cuarentena(Mundo, Individuo, mod_params, ind_params)
modelo.correr(dias_de_simulacion, show=True)
modelo.guardar(carpeta_de_salida+'cuarentena_simple1.pk')

##Caso Adaptable
print('Adaptable')
carpeta_de_salida = 'resultadosCasos/Adaptable/'
nuevos_params = {'umbral_rest': 0.005,
                 'umbral-rest': 0.01,
                 'alpha': 0}
ind_params, mod_params = modificar_parametros(ind_params, mod_params,
                                                nuevos_params)
modelo = Adaptable(Mundo, Individuo, mod_params, ind_params)
modelo.correr(dias_de_simulacion, show=True)
modelo.guardar(carpeta_de_salida+'adaptable_simple1.pk')
"""