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
from utils import convertir_corrida, leer_historico, calcular_error


class Visualizador():
    def __init__(self, path, figsize = (15,8)):
        plt.ion()
        self.fig = plt.figure(constrained_layout = True,
                              figsize = figsize)

        with open('Datos/datos.pk', 'rb') as f:
            regiones = pk.load(f)
            self.nom_mun = list(regiones.keys())
            self.posiciones = {k:norm_coord(regiones[k]['centro']) for k in regiones}
            self.tamanos = {k:max(np.log(regiones[k]['pob'])**2.5,20) for k in regiones}

        
        self.ind_x_agente = 5
        self.datos = convertir_corrida(path)

        self.dias = self.datos.index
        self.hist = leer_historico(solo_activos = True,
                              intervalo = (self.dias[0], self.dias[-1]),
                              ind_x_agente = self.ind_x_agente)

        self.region_part = 'Mérida'
        self.t = 0
        self.t_final = len(self.dias)
        self.max_inf = 1 #Número máximo de infectados, se actualiza más tarde
        
        gs = self.fig.add_gridspec(2,4)
        self.ax_mapa = self.fig.add_subplot(gs[:2,:2])
        self.ax_gen  = self.fig.add_subplot(gs[0,2:])
        self.ax_part = self.fig.add_subplot(gs[1,2:])
        
        ## Se definen los widgets
        ### Slider
        slider_ax = plt.axes([0.06,0.04,0.3,0.02])
        self.slider = Slider(slider_ax, 'Paso', 0, self.t_final-1,
                             valinit = self.t, valstep = 1, valfmt='%3i')
        self.slider.on_changed(self.update)

        self.init_plot()
        
    def init_plot(self):
        estado = self.obtener_estado_mapa(0)
        self.mapa_scatter = self.ax_mapa.scatter(estado[:,0], estado[:,1],
                                   s = estado[:,2], c = estado[:,3],
                                   #cmap = plt.get_cmap('jet')
                                   )
        self.ax_mapa.axis('off')
        cursor_mapa = mplcursors.cursor(self.mapa_scatter)#, hover=True)
        @cursor_mapa.connect("add")
        def _(sel):
            tar_nom = self.nom_mun[sel.target.index]
            datos_sel = self.datos[tar_nom].loc[self.dias[self.t]]
            #sel.annotation.get_bbox_patch().set(fc="white")
            #sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)
            sel.annotation.set_text(f'{tar_nom}\n'
                                    f'S: {int(datos_sel["S"])}\n'
                                    f'E: {int(datos_sel["E"])}\n'
                                    f'I: {int(datos_sel["I"])}\n'
                                    f'R: {int(datos_sel["R"])}')

        intervalo_hist = self.hist.columns.get_level_values(0)#np.arange(0, self.hist.shape[1])
  
        ## General
        datos_hist = self.hist.sum(axis=0)
        self.scat_gen = {'S':None, 'E':None, 'I':None, 'R':None, 'Activos':None}
        for val in self.scat_gen.keys():
            if val in ['E','I']:
                datos_gen = self.datos.iloc[:,
                            self.datos.columns.get_level_values(1)==val].sum(axis=1)
                if val=='I': 
                    self.max_inf = datos_gen.max()
                    print(f'El máximo de infectados es {self.max_inf}')
                self.scat_gen[val] = self.ax_gen.plot_date(self.dias,
                                                            datos_gen,
                                                            fmt = '-',
                                                            label = val)
            elif val == 'Activos':
                self.scat_gen[val] = self.ax_gen.plot_date(intervalo_hist,
                                                            datos_hist,
                                                            fmt='-',
                                                            label=val)
        self.ax_gen.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        self.ax_gen.set_xlabel(f'Días a partir del día cero: {self.dias[0].strftime("%d/%m")}')
        self.ax_gen.set_ylabel(f'Agentes ({self.ind_x_agente} individuos reales por agente)')
        self.gen_vline = self.ax_gen.axvline(self.ax_gen.get_xticks()[0])
        self.ax_gen.set_title('Avance general')
        
        ##Particular
        datos_part = self.datos[self.region_part]#[0]
        hist_part = self.hist.loc[self.region_part]
        self.scat_part = {'S':None, 'E':None, 'I':None, 'R':None, 'Activos':None}
        for val in self.scat_part.keys():
            if val in ['E','I']:
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
        self.ax_part.set_xlabel(f'Días a partir del día cero: {self.dias[0].strftime("%d/%m")}')
        self.ax_part.set_ylabel(f'Agentes ({self.ind_x_agente} individuos reales por agente)')
        self.part_vline = self.ax_part.axvline(self.ax_gen.get_xticks()[0])
        self.ax_part.set_title(f'Avance del nodo {self.region_part}')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.show(block = True)
    
    def update(self,i):
        i = int(i)
        self.t = i
        self.update_map(i)
        self.update_gen(i)
        self.update_part(i)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    
    def update_gen(self, i):
        self.gen_vline.set_xdata([self.dias[i],self.dias[i]])
    
    def update_part(self, i):
        self.part_vline.set_xdata([self.dias[i],self.dias[i]])

    def update_map(self, i):
        #I = []
        #for region in self.nom_mun:
        #    infectados = self.datos[region].iloc[i, :]['I']
        #    I.append(infectados/self.max_inf*100)
        state = self.obtener_estado_mapa(i)
        self.mapa_scatter.set_offsets(state[:,:2])
        self.mapa_scatter.set_sizes(state[:,2])
        self.mapa_scatter.set_array(state[:,3])
        
        
    def obtener_estado_mapa(self, step):
        estado = np.zeros((len(self.nom_mun), 4)) #x, y, tamaño, color
        for j, region in enumerate(self.nom_mun):
            coord = self.posiciones[region]
            infectados = self.datos[region].iloc[step, :]['I']/self.max_inf
            estado[j] = np.array([coord[0], coord[1],
                                  self.tamanos[region], infectados])

        return(estado)

"""
    def leer_corrida(path):
        corrida = pd.read_pickle(path)
        totales = corrida.iloc[:,:5].values

        n_fil = corrida.shape[0]
        n_col = corrida.shape[1]
        conteos = np.zeros((n_fil, (n_col-5)*4), dtype = np.uint)
        for i in range(n_fil):
            for j in range(5,n_col-5):
                conteos[i, j*4:(j+1)*4] = np.array(corrida.iloc[i, j])
        #print(list(corrida.iloc[:,5:].columns))
        conteos_cols = pd.MultiIndex.from_product([list(corrida.iloc[:,5:].columns),
                                                  ['S','E','I','R']])
        totales_cols = pd.MultiIndex.from_product([['Total'],
                                                  ['Dia','S','E','I','R']])
        df_tot = pd.DataFrame(totales, columns = totales_cols)
        #print(df_tot)
        df_cont= pd.DataFrame(conteos, columns = conteos_cols)
        datos = pd.concat((df_tot, df_cont), axis = 1)
        self.datos = datos.groupby([('Total', 'Dia')]).mean()
"""        
        

def norm_coord(coord):
    coord = np.array(coord)[::-1]
    esq1 = np.array([-89.891994, 20.160681])
    esq2 = np.array([-88.986827, 21.288735])
    delta = esq2-esq1
    return (coord-esq1)/delta


if __name__=='__main__':
    import sys
    path = sys.argv[1]
    print(f'Leyendo archivo en {path}')
    vis = Visualizador(path)
