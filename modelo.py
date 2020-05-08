#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 20:50:13 2020

@author: carlos
"""

from Ambiente.ambiente import Mundo
from Individuos.individuo import Individuo_2

from mesa import Model
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from collections import OrderedDict
from random import choices, seed, shuffle
from math import sqrt
import pickle as pk
import numpy as np
seed(920204)

class Modelo(Model):
    #Algunas constantes
    SUCEPTIBLE = 0
    EXPUESTO = 1
    INFECTADO = 2
    RECUPERADO = 3
    salud_to_str={0:'Suceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}
    pp_dia = 1 ## Son los pasos dados por dia simulado
    def __init__(self,world_object, agent_object, ind_attrs):
        super().__init__()
        self.mundo = world_object(self,agent_object)
        self.ind_attrs = ind_attrs
        self.schedule = RandomActivation(self)
        self.generar_regiones()
        self.n_paso = 0
        
        ## Se define el grid que se representarÃ¡ en la 
        #self.grid = self.ciudad.nodes['ciudad']['espacio']
        model_reporters = {'Suceptibles': self.conteo_func(self.SUCEPTIBLE),
                            'Expuestos': self.conteo_func(self.EXPUESTO),
                            'Infectados': self.conteo_func(self.INFECTADO),
                            'Recuperados': self.conteo_func(self.RECUPERADO)}
        reg_reporters = {k: self.conteo_por_reg(k) for k in self.regiones}
        self.datacollector = DataCollector({**model_reporters, **reg_reporters})
        self.conteo_instantaneo = self.conteo()
    
    def generar_regiones(self):
        datos = self.leer_regiones('Datos/datos.pk')
        conexiones = self.generar_lista_de_aristas('Datos/conexiones.csv',
                                                   list(datos.keys()))
        seleccionadas = ('Progreso,Ucú,Umán,Chocholá,Akil,Tekax,'+
                         'Mérida,Abalá,Tecoh,Timucuy,Acanceh,Kanasín,Tixpéhual,Tixkokob,Maní,Chacsinkín,'+
                         'Yaxkukul,Concal,Sacalum,Chapab,Seyé,Oxkutzcab').split(',')
        #print(seleccionadas)
        ids_start=0
        self.regiones = {}
        for region in datos:
            if region not in seleccionadas: continue
            self.regiones[region] = datos[region]
            tamano = 20
            pob = self.regiones[region]['pob']//100
            n_infectados = 1
            ids = [i for i in range(ids_start, ids_start+pob)]
            ids_start += pob
            individuos = self.mundo.generar_individuos(pob,
                                                       ids = ids,
                                                       attrs= self.ind_attrs)

            print(f'{region}: {pob} agentes, {len(individuos)} creados')
            for ind in individuos:
                if n_infectados>0:
                    ind.salud = self.INFECTADO
                    n_infectados -= 1
                    
                self.schedule.add(ind)
            
            self.mundo.crear_nodo(region, 'region',
                                  ocupantes = individuos,
                                  tamano = tamano,
                                  ind_pos_def = 'aleatorio'
                                  )
        self.mundo.add_weighted_edges_from(conexiones, weight = 'peso')
        posiciones = {k: list(self.regiones[k]['centro'])[::-1] for k in self.regiones}
        self.mundo.visualizar(pos = posiciones, with_labels = True)
        

    def norm_coord(self, coord):
        coord = np.array(coord)
        esq1 = np.array([21.670833, -90.621414])
        esq2 = np.array([19.546208, -87.449881])
        delta = esq2-esq1
        return (coord-esq1)/delta
        
    
    def obtener_rectangulo(self, regiones):
        latlim = [-1e100, 1e100]
        lonlim = [-1e100, 1e100]
        for region in regiones:
            print(region)
            for coord in regiones[region]['limites'][0]:
                latlim = [min(latlim[0], coord[0]), max(latlim[1], coord[0])]
                lonlim = [min(lonlim[0], coord[1]), max(lonlim[1], coord[1])]
        return latlim, lonlim
   
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
        with open(path, 'r') as f:
            lines = f.readlines()
            assert len(lines)==len(regiones), f'{len(lines)}!={len(regiones)}'
            for i, line in enumerate(lines):
                datos = line.split(',')
                assert len(datos)==len(regiones)
                
                conexiones += [(regiones[i], regiones[j], float(x.strip()))\
                               for j, x in enumerate(line.split(',')) if x.strip()!='0.0']
        return conexiones
    
    def obtener_infectados(self, path):
        infectados = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                region, n_infectados = line.strip('\n').split(',')
                infectados[region] = int(n_infectados)
        return infectados
                
    def step(self):
        self.momento = self.n_paso % self.pp_dia #es el momento del dia
        self.conteo()
        self.datacollector.collect(self)
        self.schedule.step()
        self.n_paso += 1
        
        
    def correr(self, n_steps):
        bloques = int(n_steps*0.1)
        print('---- Corriendo simulación ----')
        for i in range(n_steps):
            self.step()
            if int(i%bloques) == 0:
                print('%d%% ... '%(int(i/n_steps*100)), end = '')
        print('100%')

            

attrs_individuos = {#De comportamiento
                    'evitar_agentes': False,
                    'evitar_sintomaticos': False,
                    'distancia_paso': 1,
                    'prob_movimiento':0.5,
                    'prob_mov_nodos':0.1,
                    'activar_cuarentena': False,
                    'quedate_en_casa': False,
                    #Ante la enfermedad
                    'prob_contagiar': 0.2,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }

modelo = Modelo(Mundo, Individuo_2,
                attrs_individuos)

modelo.mundo.ver_espacio('Mérida')
modelo.correr(120)
data = modelo.datacollector.get_model_vars_dataframe()
data.to_pickle('Corridas/corrida1.pk')
data[['Suceptibles', 'Expuestos', 'Infectados', 'Recuperados']].plot()