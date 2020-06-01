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
import matplotlib.pyplot as plt

from utils import obtener_movilidad, leer_historico

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
        self.movilidad = obtener_movilidad()
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
        reg_reporters = {k: self.conteo_por_reg(k) for k in self.regiones}
        self.datacollector = DataCollector({**model_reporters, **reg_reporters})
        self.conteo_instantaneo = self.conteo()

    
    def generar_regiones(self):
        datos = self.leer_regiones('Datos/datos.pk')
        conexiones = self.generar_lista_de_aristas('Datos/adyacencia.pk',
                                                   list(datos.keys()))
        #infectados = self.obtener_infectados('Datos/infectados.csv',
        #                                     list(datos.keys()))
        #historico = leer_historico()
        #infectados = historico[(self.dia_cero, 'Activos')]
        #fecha = self.params['dia_cero']################################3
        ids_start=0
        self.regiones = {}
        for region in datos:
            #if region not in seleccionadas: continue
            self.regiones[region] = datos[region]
            tamano = self.params['area']
            pob = ceil(self.regiones[region]['pob']//self.params['inds_x_agente'])
            ids = [i for i in range(ids_start, ids_start+pob)]
            ids_start += pob

            individuos = self.mundo.generar_individuos(pob,
                                                       ids = ids,
                                                       attrs= self.ind_attrs)
            """
            n_infectados = ceil(infectados[region]/self.params['inds_x_agente'])\
                            if infectados.get(region, None) is not None else 0
            n_susceptibles = ceil(n_infectados*self.prop_inf_exp)
            #print(f'{region}: {pob} agentes, {n_infectados} infectados, {n_susceptibles} susceptibles')
            """
            if region=='Mérida':
                for i in range(self.params['expuestos_iniciales']):
                    individuos[i].salud=self.EXPUESTO

            for ind in individuos:
                """
                if n_infectados>0:
                    ind.salud = self.INFECTADO
                    n_infectados -= 1
                    #print(f'\tSe agrega un infectado ind {ind.unique_id}')
                elif n_infectados==0 and n_susceptibles>0:
                    ind.salud = self.EXPUESTO
                    n_susceptibles -= 1
                    #print(f'\tSe agrega un expuesto ind {ind.unique_id}')
                """
                self.schedule.add(ind)

            #print(f'La región {region} tiene {len(individuos)}')
            self.mundo.crear_nodo(region, 'municipio',
                                  ocupantes = individuos,
                                  tamano = tamano,
                                  ind_pos_def = 'aleatorio'
                                  )
        #print('Se crean las aristas')
        self.mundo.add_weighted_edges_from(conexiones, weight = 'peso')
        #print([a.salud for a in self.schedule.agents if a.salud==self.EXPUESTO])
        #posiciones = {k: list(self.regiones[k]['centro'])[::-1] for k in self.regiones}
        #self.mundo.visualizar(pos = posiciones, with_labels = True)
        

    def norm_coord(self, coord):
        coord = np.array(coord)
        esq1 = np.array([21.670833, -90.621414])
        esq2 = np.array([19.546208, -87.449881])
        delta = esq2-esq1
        return (coord-esq1)/delta
        
   
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

if __name__=='__main__':
    from Ambiente.ambiente import Mundo
    from Individuos.individuo import Individuo_2
    import datetime

    dia_cero = datetime.datetime(2020,3,10)
    un_dia = datetime.timedelta(days = 1)
    dia_final = datetime.datetime(2020,3,20)#list(hist.columns.get_level_values(0))[-1]
    n_dias = int((dia_final-dia_cero)/un_dia)+1

    atributos = {#De comportamiento
                    'evitar_agentes': True,
                    'distancia_paso': 1,
                    'prob_movimiento':0.4,
                    'frac_mov_nodos':0.2,
                    #Ante la enfermedad
                    'prob_contagiar': 0.25,
                    'prob_infectarse': 0.4,
                    'radio_de_infeccion': 1
                    }
    modelo_params = {
                    'area':5,
                    'inds_x_agente':500,
                    'dia_cero':dia_cero,
                    'expuestos_iniciales':5
                }

    modelo = Modelo(Mundo, Individuo_2,
                modelo_params,
                atributos,
                rand_seed = 920204)
    modelo.correr(n_dias*4, show=True)
    modelo.plot(['Expuestos', 'Infectados'])
