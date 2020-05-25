from mesa import Model
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from collections import OrderedDict
from random import choices, seed, shuffle
from math import sqrt, ceil
import pickle as pk
import pandas as pd
import numpy as np
import datetime

from utils import obtener_movilidad, leer_historico

class Modelo(Model):
    #Algunas constantes
    SUSCEPTIBLE = 0
    EXPUESTO = 1
    INFECTADO = 2
    RECUPERADO = 3
    salud_to_str={0:'Susceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}
    pp_dia = 4 ## Son los pasos dados por dia simulado

    def __init__(self,world_object, agent_object, params, ind_attrs):
        super().__init__()
        self.params = params
        self.mundo = world_object(self,agent_object)
        self.movilidad = obtener_movilidad()
        self.dia_cero = params['dia_cero']
        self.prop_inf_suscep = params['prop_inf_suscep'] #Proporcion entre infectaros y suceptibles a la fecha
        self.un_dia = datetime.timedelta(days=1)
        self.ind_attrs = ind_attrs
        self.schedule = RandomActivation(self)
        self.generar_regiones()
        self.dia = 0
        self.n_paso = 0
        
        ## Se define el grid que se representarÃ¡ en la 
        #self.grid = self.ciudad.nodes['ciudad']['espacio']
        model_reporters = { 'Dia': lambda x: x.dia,
                            'Suceptibles': self.conteo_func(self.SUSCEPTIBLE),
                            'Expuestos': self.conteo_func(self.EXPUESTO),
                            'Infectados': self.conteo_func(self.INFECTADO),
                            'Recuperados': self.conteo_func(self.RECUPERADO)}
        reg_reporters = {k: self.conteo_por_reg(k) for k in self.regiones}
        self.datacollector = DataCollector({**model_reporters, **reg_reporters})
        self.conteo_instantaneo = self.conteo()

    
    def generar_regiones(self):
        datos = self.leer_regiones('Datos/datos.pk')
        conexiones = self.generar_lista_de_aristas('Datos/adyacencia.pk',
                                                   list(datos.keys()))
        #infectados = self.obtener_infectados('Datos/infectados.csv',
        #                                     list(datos.keys()))
        historico = leer_historico()
        infectados = historico.loc[:, (self.dia_cero, 'Activos')]
        fecha = self.params['dia_cero']################################3
        ids_start=0
        self.regiones = {}
        for region in datos:
            #if region not in seleccionadas: continue
            self.regiones[region] = datos[region]
            tamano = self.params['area']
            pob = ceil(self.regiones[region]['pob']//self.params['inds_x_agente'])
            ids = [i for i in range(ids_start, ids_start+pob)]
            ids_start += pob

            #ind_attrs = {**self.ind_attrs, **{'nodo_actual': region,
            #                              'nodo_casa': region}}
            individuos = self.mundo.generar_individuos(pob,
                                                       ids = ids,
                                                       attrs= self.ind_attrs)
            n_infectados = ceil(infectados[region]/self.params['inds_x_agente'])\
                            if infectados.get(region, None) is not None else 0
            n_susceptibles = ceil(n_infectados*self.prop_inf_suscep)
            #print(f'{region}: {pob} agentes, {n_infectados} infectados, {n_susceptibles} susceptibles')
            for ind in individuos:
                if n_infectados>0:
                    ind.salud = self.INFECTADO
                    n_infectados -= 1
                if n_infectados==0 and n_susceptibles>0:
                    ind.salud = self.SUSCEPTIBLE
                    n_susceptibles -= 1

                self.schedule.add(ind)
            
            self.mundo.crear_nodo(region, 'municipio',
                                  ocupantes = individuos,
                                  tamano = tamano,
                                  ind_pos_def = 'aleatorio'
                                  )
        #print('Se crean las aristas')
        self.mundo.add_weighted_edges_from(conexiones, weight = 'peso')
        posiciones = {k: list(self.regiones[k]['centro'])[::-1] for k in self.regiones}
        #self.mundo.visualizar(pos = posiciones, with_labels = True)
        

    def norm_coord(self, coord):
        coord = np.array(coord)
        esq1 = np.array([21.670833, -90.621414])
        esq2 = np.array([19.546208, -87.449881])
        delta = esq2-esq1
        return (coord-esq1)/delta
        
   
    def conteo(self):
        #Una funciÃ³n para contar los casos actuales en la ciudad
        self.conteo_instantaneo = [0,0,0,0]
        for a in self.schedule.agents:
            self.conteo_instantaneo[a.salud] += 1
        return self.conteo_instantaneo

    def conteo_func(self, tipo):
        def contar(modelo):
            return modelo.conteo_instantaneo[tipo]
        return contar
    
    def conteo_por_reg(self, reg):
        def contar(modelo):
            ags = modelo.mundo.obtener_agentes_en_nodo(reg)
            conteo = [0,0,0,0]
            for a in ags:
                conteo[a.salud]+=1
            return conteo
        return contar

    def leer_regiones(self,path):
        with open(path, 'rb') as f:
            datos = pk.load(f)
        return datos
    
    def generar_lista_de_aristas(self, path, regiones):
        conexiones = []
        with open(path, 'rb') as f:
            datos = pk.load(f)
            assert len(datos)==len(regiones), f'{len(datos)}!={len(regiones)}'
            a_agregar=[]
            for region in datos:                
                a_agregar = [(region, nueva, peso)\
                               for nueva, peso in datos[region]]
                conexiones.extend(a_agregar)
        return conexiones
    
    def obtener_infectados(self, path, regiones):
        infectados = {}
        with open(path, 'r') as f:
            for line in f.readlines()[4:]:
                datos= line.split(',')
                if datos[1] not in regiones:
                    print(f'{datos[1]} no está en regiones')
                else:
                    infectados[datos[1]] = int(datos[5])
        return infectados
                
    def step(self):
        self.dia = self.n_paso//self.pp_dia #es el momento del dia
        self.conteo()
        self.datacollector.collect(self)
        self.schedule.step()
        self.n_paso += 1
        
        
    def correr(self, n_steps):
        bloques = int(n_steps*0.1)
        #print('---- Corriendo simulación ----')
        for i in range(n_steps):
            self.step()
            if int(i%bloques) == 0:
                #print('%d%% ... '%(int(i/n_steps*100)), end = '')
        #print('100%')

if __name__=='__main__':
    from Ambiente.ambiente import Mundo
    from Individuos.individuo import Individuo_2
    atributos = {#De comportamiento
                    'evitar_agentes': True,
                    'distancia_paso': 1,
                    'prob_movimiento':0.1,
                    'frac_mov_nodos':0.05,
                    #Ante la enfermedad
                    'prob_contagiar': 0.1,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }
    modelo_params = {
                    'area':1,
                    'inds_x_agente':500,
                    'dia_cero':datetime.datetime(2020,4,17),
                    'prop_inf_suscep': 2
                }

    modelo = Modelo(Mundo, Individuo_2,
                modelo_params,
                atributos)
    modelo.correr(10)
    data = modelo.datacollector.get_model_vars_dataframe()
    data.plot()
