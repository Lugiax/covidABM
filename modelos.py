from mesa import Model
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from collections import OrderedDict
from random import Random, shuffle
from math import sqrt, ceil
import pickle as pk
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from utils import GeneradorMovilidad, leer_historico, AnalizadorMunicipios

class Modelo(Model):
    #Algunas constantes
    SUSCEPTIBLE = 0
    EXPUESTO = 1
    INFECTADO = 2
    RECUPERADO = 3
    salud_to_str={0:'Susceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}
    pp_dia = 4 ## Son los pasos dados por dia simulado

    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__()
        self.rand = Random(rand_seed)
        self.params = params
        self.mundo = world_object(self,agent_object)
        self.movilidad = obtener_movilidad()###CAMBIAR -------------------------------------
        self.DatosMun = AnalizadorMunicipios()
        self.dia_cero = params['dia_cero']
        #self.prop_inf_exp = params['prop_inf_exp'] #Proporcion entre infectaros y suceptibles a la fecha
        self.un_dia = datetime.timedelta(days=1)
        self.ind_attrs = ind_attrs
        self.schedule = RandomActivation(self)
        self.generar_regiones()
        self.dia = 0
        self.n_paso = 0
        
        ## Se define el grid que se representarÃ¡ en la 
        #self.grid = self.ciudad.nodes['ciudad']['espacio']
        model_reporters = { 'Fecha': lambda x: x.dia_cero+x.dia*x.un_dia,
                            'Susceptibles': self.conteo_func(self.SUSCEPTIBLE),
                            'Expuestos': self.conteo_func(self.EXPUESTO),
                            'Infectados': self.conteo_func(self.INFECTADO),
                            'Recuperados': self.conteo_func(self.RECUPERADO)}
        reg_reporters = {k: self.conteo_por_reg(k) for k in self.DatosMun.municipios}
        self.datacollector = DataCollector({**model_reporters, **reg_reporters})
        self.conteo_instantaneo = self.conteo()

    
    def generar_regiones(self):
        conexiones = self.generar_lista_de_aristas('Datos/adyacencia.pk',
                                                    self.DatosMun.municipios)

        ids_start=0
        for region in self.DatosMun.municipios:
            region_num = self.DatosMun.obtener_numero(region)
            tamano = int(self.DatosMun.obtener_densidad(region_num))*4
            n_pob = ceil(self.DatosMun.obtener_poblacion(region_num)//self.params['inds_x_agente'])
            
            ids = [i for i in range(ids_start, ids_start+n_pob)]
            ids_start += n_pob

            individuos = self.mundo.generar_individuos(n_pob,
                                                       ids = ids,
                                                       attrs= self.ind_attrs)

            if region=='Mérida':
                for i in range(self.params['expuestos_iniciales']):
                    individuos[i].salud=self.EXPUESTO

            for ind in individuos:
                self.schedule.add(ind)

            print(f'La región {region} de tamaño {tamano} tiene {n_pob}')
            self.mundo.crear_nodo(region, 'municipio',
                                  ocupantes = individuos,
                                  tamano = tamano,
                                  ind_pos_def = 'aleatorio'
                                  )
        self.mundo.add_weighted_edges_from(conexiones, weight = 'peso')
        
   
    def conteo(self):
        #Una función para contar los casos actuales en la ciudad
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
         
    def step(self):
        self.dia = self.n_paso//self.pp_dia #es el momento del dia

        if self.dia == 4:
            ##En el cuarto día, que corresponde al primer caso en
            #Yucatán, se planta un infectado. Esto para asegurar que
            #siempre habrá un infectado
            agentes = self.mundo.obtener_agentes_en_nodo('Mérida')
            agentes[0].salud = self.INFECTADO

        self.conteo()
        self.datacollector.collect(self)
        self.schedule.step()
        self.n_paso += 1
        
        
    def correr(self, n_steps, show=False):
        bloques = int(n_steps*0.1)
        if show:print('---- Corriendo simulación ----')
        for i in range(n_steps):
            self.step()
            if show and int(i%bloques) == 0:
                print('%d%% ... '%(int(i/n_steps*100)), end = '\n')
        if show: print('100%')

    def plot(self, names = None):
        data = self.datacollector.get_model_vars_dataframe()
        if names is None:
            data.plot()
        elif isinstance(names, list):
            data[names].plot()
        else:
            print('Se debe ingresar los nombres de las columnas en una lista')
        plt.show(block=True)



class Simple:
    #Algunas constantes
    SUSCEPTIBLE = 0
    EXPUESTO = 1
    INFECTADO = 2
    RECUPERADO = 3
    salud_to_str={0:'Susceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}
    pp_dia = 2 ## Son los pasos dados por dia simulado

    def __init__(self,world_object, agent_object, params, ind_attrs, rand_seed=None):
        super().__init__()
        self.rand = Random(rand_seed)
        self.random = self.rand
        self.params = params
        self.ind_attrs = ind_attrs
        self.mundo = world_object(self,agent_object)
        self.movilidad = GeneradorMovilidad(semanas_a_agregar = 8,
                                            valor_de_relleno = params['p_reduccion_mov'])
        self.DatosMun = AnalizadorMunicipios()
        #self.prop_inf_exp = params['prop_inf_exp'] #Proporcion entre infectaros y suceptibles a la fecha
        self.un_dia = datetime.timedelta(days=1)
        self.schedule = RandomActivation(self)

        self.conteo_instantaneo = [0,0,0,0]
        self.generar_region()
        self.dia = 0
        self.fecha = params['dia_cero']
        self.n_paso = 0
        
        model_reporters = { 'Fecha': lambda x: x.fecha,
                            'Susceptibles': self.conteo_func(self.SUSCEPTIBLE),
                            'Expuestos': self.conteo_func(self.EXPUESTO),
                            'Infectados': self.conteo_func(self.INFECTADO),
                            'Recuperados': self.conteo_func(self.RECUPERADO)}
        self.datacollector = DataCollector(model_reporters)
        self.conteo_instantaneo = self.conteo()

    
    def generar_region(self):
        n_pob = ceil(self.DatosMun.obtener_poblacion()//self.params['inds_x_agente'])
        individuos = self.mundo.generar_individuos(n_pob,
                                                   attrs= self.ind_attrs)
        individuos[0].salud = self.INFECTADO

        for ind in individuos:
            if self.params['expuestos_iniciales']>0 and ind.salud != self.INFECTADO:
                ind.salud = self.EXPUESTO
                self.params['expuestos_iniciales']-=1
            self.schedule.add(ind)

        self.mundo.crear_nodo('Yucatán', 'municipio',
                              ocupantes = individuos,
                              tamano = self.params['tamano'],
                              ind_pos_def = 'aleatorio'
                              )
        
   
    def conteo(self):
        #Una función para contar los casos actuales en la ciudad
        self.conteo_instantaneo = [0,0,0,0]
        for a in self.schedule.agents:
            self.conteo_instantaneo[a.salud] += 1
        return self.conteo_instantaneo

    def conteo_func(self, tipo):
        def contar(modelo):
            return modelo.conteo_instantaneo[tipo]
        return contar
    
    def step(self):
        self.dia = self.n_paso//self.pp_dia
        self.fecha = self.params['dia_cero']+ self.un_dia*self.dia
        self.semana = self.fecha.isocalendar()[1]
        self.porcentaje_movilidad = 1 + self.movilidad.generar(self.semana)/100

        self.conteo()

        self.datacollector.collect(self)
        self.schedule.step()
        self.n_paso += 1
        
        
    def correr(self, n_steps, show=False):
        bloques = int(n_steps*0.1)

        if show:print(f'---- Corriendo simulación ----\n#Agentes: {len(self.schedule.agents)}')
        t0 = time.time()
        for i in range(n_steps):
            self.step()
            if show and int(i%bloques) == 0:
                ti = time.time()
                print(f'{int(i/n_steps*100)}% ... semana {self.semana}, '
                    f'{int(self.porcentaje_movilidad*100)}% de movilidad global. Tiempo {(ti-t0)/60:.2} minutos')
        if show: print('100%')

    def plot(self, names = None):
        data = self.datacollector.get_model_vars_dataframe()
        data = data.groupby('Fecha').mean()
        if names is None:
            data.plot()
        elif isinstance(names, list):
            data[names].plot()
        else:
            print('Se debe ingresar los nombres de las columnas en una lista')
        plt.show(block=True)

    def devolver_df(self):
        return self.datacollector.get_model_vars_dataframe()

    def guardar(self, path = 'corrida.pk'):
        modelo.datacollector.get_model_vars_dataframe().to_pickle(path)




if __name__=='__main__':
    from Ambiente.ambiente import Mundo
    from Individuos.individuo import Individuo
    import datetime
    from utils import GraficadorSimple

    dia_cero = datetime.datetime(2020,3,10)
    un_dia = datetime.timedelta(days = 1)
    dia_final = datetime.datetime(2020,3,20)#list(hist.columns.get_level_values(0))[-1]
    n_dias = int((dia_final-dia_cero)/un_dia)+1

    atributos = {#De comportamiento
                    'evitar_agentes': False,
                    'distancia_paso': 1,
                    'prob_movimiento':1,
                    #Ante la enfermedad
                    'prob_contagiar': 0.5,
                    'prob_infectarse': 0.1,
                    'radio_de_infeccion': 1
                    }
    modelo_params = {
                    'inds_x_agente':1000,
                    'tamano':50,
                    'dia_cero':dia_cero,
                    'expuestos_iniciales':5,
                    'p_reduccion_mov': 0
                }

    modelo = Simple(Mundo, Individuo,
                modelo_params,
                atributos,
                rand_seed = 920204)
    modelo.correr(500, show=True)
    datos = modelo.devolver_df()
    graf = GraficadorSimple(datos)
    graf.graficar()
    #modelo.plot(['Expuestos', 'Infectados'])
