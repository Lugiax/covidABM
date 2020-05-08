#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:51:33 2020

@author: carlos
"""

import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


class Visualizador():
    def __init__(self, datos, posiciones, tamanos, figsize = (15,8)):
        plt.ion()
        self.fig = plt.figure(constrained_layout = True,
                              figsize = figsize)
        gs = self.fig.add_gridspec(2,4)
        self.ax_mapa = self.fig.add_subplot(gs[:2,:2])
        self.ax_gen  = self.fig.add_subplot(gs[0,2:])
        self.ax_part = self.fig.add_subplot(gs[1,2:])
        
        self.edos_salud = 'Suceptibles','Expuestos','Infectados','Recuperados'
        self.datos = datos
        self.posiciones = posiciones
        self.tamanos = tamanos
        self.tamano_ventana = 50
        self.region_part = 'Akil'
        
        self.init_plot()
        

    def init_plot(self):
        estado = self.obtener_estado_mapa(0)
        self.mapa_scatter = self.ax_mapa.scatter(estado[:,0], estado[:,1],
                                   s = estado[:,2], c = estado[:,3],
                                   cmap = plt.get_cmap('jet'))
        
        ## General
        datos_gen = self.datos['Total'][0]
        self.scat_gen = {'Suceptibles':None, 'Expuestos':None, 'Infectados':None, 'Recuperados':None}
        for t,val in zip(self.scat_gen.keys(),
                         datos_gen):
            self.scat_gen[t] = self.ax_gen.plot(0, val, label= t)
        self.ax_gen.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        self.ax_gen.set_xlim(0,self.tamano_ventana)
        self.ax_gen.set_ylim(0,datos_gen.max())
        
        ##Particular
        datos_part = self.datos[self.region_part][0]
        self.scat_part = {'Suceptibles':None, 'Expuestos':None, 'Infectados':None, 'Recuperados':None}
        for t,val in zip(self.scat_part.keys(),
                         datos_part):
            self.scat_part[t] = self.ax_part.plot(0, val, label= t)
        self.ax_part.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        self.ax_part.set_xlim(0,self.tamano_ventana)
        self.ax_part.set_ylim(0,datos_part.max())
        
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update(self,i):
        self.update_map(i)
        self.update_gen(i)
        self.update_part(i)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    
    def update_gen(self, i):
        t_inicio = max(0,i-self.tamano_ventana)
        t_fin = max(self.tamano_ventana, i)
        datos_gen = self.datos['Total'][:i]
        for j, tipo in enumerate(self.edos_salud):
            self.scat_gen[tipo][0].set_data(np.arange(i),
                                   datos_gen[:, j])
        self.ax_gen.set_xlim(t_inicio,t_fin)
    
    def update_part(self, i):
        t_inicio = max(0,i-self.tamano_ventana)
        t_fin = max(self.tamano_ventana, i)
        print(i)
        datos_part = self.datos[self.region_part][:i]
        for j, tipo in enumerate(self.edos_salud):
            print(j, datos_part)
            self.scat_part[tipo][0].set_data(np.arange(i),
                                   datos_part[:, j])
        self.ax_part.set_xlim(t_inicio,t_fin)
        self.ax_part.set_ylim(0,datos_part.max())

    def update_map(self, i):
        state = self.obtener_estado_mapa(i)
        self.mapa_scatter.set_offsets(state[:,:2])
        self.mapa_scatter.set_sizes(state[:,2])
        self.mapa_scatter.set_array(state[:,3])
        
        
    def obtener_estado_mapa(self, step):
        estado = np.zeros((len(regiones), 4)) #x, y, tamaño, color
        for j, region in enumerate(self.posiciones):
            coord = self.posiciones[region]
            datos = self.datos[region][step]
            infectados = datos[2]/datos.sum()
            estado[j] = np.array([coord[0], coord[1],
                                  self.tamanos[region], infectados])
        return(estado)
        

def norm_coord(coord):
    coord = np.array(coord)[::-1]
    esq1 = np.array([-89.891994, 20.160681])
    esq2 = np.array([-88.986827, 21.288735])
    delta = esq2-esq1
    return (coord-esq1)/delta

seleccionadas = ('Progreso,Ucú,Umán,Chocholá,Akil,Tekax,'+
                'Mérida,Abalá,Tecoh,Timucuy,Acanceh,Kanasín,Tixpéhual,Tixkokob,Maní,Chacsinkín,'+
                'Yaxkukul,Concal,Sacalum,Chapab,Seyé,Oxkutzcab').split(',')
corrida = pd.read_pickle('Corridas/corrida1.pk')
with open('Datos/datos.pk', 'rb') as f:
    regiones = pk.load(f)
    posiciones = {k:norm_coord(regiones[k]['centro']) for k in regiones if k in seleccionadas}
    tamanos = {k:max(np.log(regiones[k]['pob'])**2.5,20) for k in regiones if k in seleccionadas}
n_it = corrida.shape[0]
datos={}
datos['Total'] = corrida.iloc[:,:4].values
for col in list(corrida.iloc[:,4:].columns):
    valores = np.zeros((n_it,4), dtype = np.uint)
    for i in range(n_it):
        valores[i] = np.array(corrida.loc[i, col])
    datos[col] = valores

vis = Visualizador(datos, posiciones, tamanos)
for i in range(1,corrida.shape[0]):
    vis.update(i)
    sleep(0)


    
    
    
    
