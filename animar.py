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
    def __init__(self, corrida, posiciones, tamanos, figsize = (5,5)):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize = figsize)
        self.corrida = corrida
        self.posiciones = posiciones
        self.tamanos = tamanos
        estado_init = self.obtener_estado(0)
        self.show(estado_init)

    def show(self, state, title = ''):
        self.vis = self.ax.scatter(state[:,0], state[:,1],
                                   s = state[:,2], c = state[:,3],
                                   cmap = plt.get_cmap('jet'))

        self.ax.set_title(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, i, title = ''):
        state = self.obtener_estado(i)
        self.vis.set_offsets(state[:,:2])
        self.vis.set_sizes(state[:,2])
        self.vis.set_array(state[:,3])
        self.ax.set_title(title)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def obtener_estado(self, step):
        estado = np.zeros((len(regiones), 4)) #x, y, tamaño, color
        for j, region in enumerate(self.posiciones):
            coord = self.posiciones[region]
            datos = self.corrida.loc[step, region]
            infectados = datos[2]/sum(datos)
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

vis = Visualizador(corrida, posiciones, tamanos)
for i in range(1,corrida.shape[0]):
    vis.update(i)
    sleep(0.2)


    
    
    
    
