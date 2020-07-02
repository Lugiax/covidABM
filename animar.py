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
from matplotlib.widgets import Slider, Button, TextBox
import mplcursors
import datetime
from collections import OrderedDict
from utils import convertir_corrida, leer_historico, calcular_error, AnalizadorMunicipios
import os


class Visualizador():
    def __init__(self, path, ver=['S','E','I','R'], ind_x_agente = 10, figsize = (15,8)):
        plt.ion()
        self.fig = plt.figure(constrained_layout = True,
                              figsize = figsize)
        
        self.fig.suptitle(f'{os.path.split(path)[-1]}',
                            fontsize=16)
        self.ver = ver

        self.DatosMun = AnalizadorMunicipios()
        self.nom_mun = self.DatosMun.municipios
        self.posiciones = self.DatosMun.obtener_coordenadas()
        self.tamanos = self.DatosMun.obtener_densidades()
        
        self.ind_x_agente = ind_x_agente
        self.datos = convertir_corrida(path)

        self.dias = self.datos.index
        self.hist = leer_historico(solo_activos = True,
                              intervalo = (self.dias[0], self.dias[-1]),
                              ind_x_agente = self.ind_x_agente)
        dias_validos = self.hist.columns.get_level_values(0)
        self.error = calcular_error(path, (dias_validos[0], dias_validos[-1]),
                                    self.ind_x_agente)
        
        self.region_part = 'Mérida'
        self.region_part_num = self.DatosMun.obtener_numero(self.region_part)
        self.fecha = self.dias[0]
        self.t_final = len(self.dias)
        self.max_inf = 1 #Número máximo de infectados, se actualiza más tarde
        
        gs = self.fig.add_gridspec(2,4)
        self.ax_mapa = self.fig.add_subplot(gs[:2,:2])
        self.ax_gen  = self.fig.add_subplot(gs[0,2:])
        self.ax_part = self.fig.add_subplot(gs[1,2:])

        self.init_plot()
        
    def init_plot(self):
        intervalo_hist = self.hist.columns.get_level_values(0)
        ## General
        datos_gen = self.datos['Total']
        datos_hist = self.hist.sum(axis=0)
        self.scat_gen = {'S':None, 'E':None, 'I':None, 'R':None, 'Activos':None}
        for val in self.scat_gen.keys():
            if val in self.ver:
                #datos_gen = self.datos.iloc[:,
                #            self.datos.columns.get_level_values(1)==val].sum(axis=1)
                if val=='I': 
                    self.max_inf = datos_gen[val].max()
                    print(f'El máximo de infectados es {self.max_inf}')
                self.scat_gen[val] = self.ax_gen.plot_date(self.dias,
                                                            datos_gen[val],
                                                            fmt = '-',
                                                            label = val)
            elif val == 'Activos':
                self.scat_gen[val] = self.ax_gen.plot_date(intervalo_hist,
                                                            datos_hist,
                                                            fmt='-',
                                                            label=val)
        self.ax_gen.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        #self.ax_gen.set_xlabel(f'Días a partir del día cero: {self.dias[0].strftime("%d/%m")}')
        self.ax_gen.set_ylabel(f'Agentes ({self.ind_x_agente} individuos reales por agente)')
        self.gen_vline = self.ax_gen.axvline(self.ax_gen.get_xticks()[0])
        self.ax_gen.set_title(f'Avance general.')# Error: {self.error.to_numpy().sum()}')
        
        ##Particular
        datos_part = self.datos[self.region_part]#[0]
        hist_part = self.hist.loc[self.region_part_num]
        self.scat_part = {'S':None, 'E':None, 'I':None, 'R':None, 'Activos':None}
        for val in self.scat_part.keys():
            if val in self.ver:
                self.scat_part[val] = self.ax_part.plot_date(self.dias,
                                                            datos_part[val],
                                                            fmt='-',
                                                            label= val)
            elif val == 'Activos':
                self.scat_part[val] = self.ax_part.plot_date(intervalo_hist,
                                                        hist_part,
                                                        fmt='-',
                                                        label = val)
        self.ax_part.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        #self.ax_part.set_xlabel(f'Días a partir del día cero: {self.dias[0].strftime("%d/%m")}')
        self.ax_part.set_ylabel(f'Agentes ({self.ind_x_agente} individuos reales por agente)')
        self.part_vline = self.ax_part.axvline(self.ax_gen.get_xticks()[0])
        self.ax_part.set_title(f'Avance del nodo {self.region_part}.')
                                #f'Error: {self.error.loc[self.region_part].sum()}')

        ## El mapa!!!!
        estado = self.obtener_estado_mapa()
        self.mapa_scatter = self.ax_mapa.scatter(estado[:,0], estado[:,1],
                                   s = estado[:,2], c = estado[:,3],
                                   vmin = 0, vmax = self.max_inf,
                                   cmap = plt.get_cmap('cool')
                                   )
        self.ax_mapa.axis('off')

        ## Se definen los widgets
        ### Slider
        slider_ax = plt.axes([0.04,0.04,0.3,0.02])
        self.slider = Slider(slider_ax, 'Paso', 0, self.t_final-1,
                             valinit = 0, valstep = 1, valfmt='%3i')
        self.slider.on_changed(self.update)
        colorbar_ax = plt.axes([0.42,0.05,0.008,0.3])
        self.fig.colorbar(self.mapa_scatter, cax = colorbar_ax,
                            label=f'#infectados ({self.ind_x_agente} ind. por agente)')

        cursor_mapa = mplcursors.cursor(self.mapa_scatter)#, hover=True)
        @cursor_mapa.connect("add")
        def _(sel):
            tar_nom = self.nom_mun[sel.target.index]
            datos_sel = self.datos[tar_nom].loc[self.fecha]
            #sel.annotation.get_bbox_patch().set(fc="white")
            #sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)
            sel.annotation.set_text(f'{tar_nom}\n'
                                    f'S: {int(datos_sel["S"])}\n'
                                    f'E: {int(datos_sel["E"])}\n'
                                    f'I: {int(datos_sel["I"])}\n'
                                    f'R: {int(datos_sel["R"])}')


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.show(block = True)
    
    def update(self,i):
        self.fecha = self.dias[int(i)]
        
        state = self.obtener_estado_mapa()
        self.mapa_scatter.set_offsets(state[:,:2])
        self.mapa_scatter.set_sizes(state[:,2])
        self.mapa_scatter.set_array(state[:,3])

        self.gen_vline.set_xdata([self.fecha,self.fecha])
        
        self.part_vline.set_xdata([self.fecha,self.fecha])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
    def obtener_estado_mapa(self):
        estado = np.zeros((len(self.nom_mun), 4)) #x, y, tamaño, color
        #import pdb; pdb.set_trace()
        for j, region in enumerate(self.nom_mun):
            region_num = self.DatosMun.obtener_numero(region)
            coord = self.posiciones.loc[region_num]
            infectados = self.datos.loc[self.fecha, (region, 'I')]
            #rint(region, coord, infectados, self.datos.loc[self.fecha,(region, 'S')])
            estado[j] = np.array([coord[1], coord[0],
                                  self.tamanos.loc[region_num], infectados])

        return(estado)

    


if __name__=='__main__':
    import sys
    path = sys.argv[1]
    print(f'Leyendo archivo en {path}')
    if sys.argv[2]:
        ver = sys.argv[2].split(',')
        vis = Visualizador(path, ver=ver)
    else:
        vis = Visualizador(path)
