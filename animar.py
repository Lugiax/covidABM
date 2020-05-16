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
from collections import OrderedDict


class Visualizador():
    def __init__(self, path, figsize = (15,8)):
        plt.ion()
        self.fig = plt.figure(constrained_layout = True,
                              figsize = figsize)
        self.leer_datos(path)
        
        self.edos_salud = 'Suceptibles','Expuestos','Infectados','Recuperados'
        self.nom_mun = list(self.datos.keys())[1:]
        self.tamano_ventana = 50
        self.region_part = 'Valladolid'
        self.t = 0
        self.t_final = self.datos['Total'].shape[0]
        
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
        
        ### Texto
        text_ax = plt.axes([0.82, 0.47, 0.1, 0.025])
        self.text_box = TextBox(text_ax, '', initial = self.region_part)
        self.text_box.on_submit(self.actualizar_nombre)
        ### Botón
        #button_ax = plt.axes([0.4,0.04,0.05,0.02])
        #self.button = Button(button_ax, 'Auto')
        #self.presionado=False
        #self.button_cid =self.button.on_clicked(self.auto_update)

        self.init_plot()
        

    def init_plot(self):
        estado = self.obtener_estado_mapa(0)
        self.mapa_scatter = self.ax_mapa.scatter(estado[:,0], estado[:,1],
                                   s = estado[:,2], c = estado[:,3],
                                   cmap = plt.get_cmap('cool'))
        cursor_mapa = mplcursors.cursor(self.mapa_scatter, hover=True)
        @cursor_mapa.connect("add")
        def _(sel):
            tar_nom = self.nom_mun[sel.target.index]
            sel.annotation.get_bbox_patch().set(fc="white")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=.5)
            sel.annotation.set_text(f'{tar_nom}\n'
                                    f'S: {self.datos[tar_nom][self.t][0]}\n'
                                    f'E: {self.datos[tar_nom][self.t][1]}\n'
                                    f'I: {self.datos[tar_nom][self.t][2]}\n'
                                    f'R: {self.datos[tar_nom][self.t][3]}')
        
        intervalo = np.arange(0,self.t_final)   
        ## General
        datos_gen = self.datos['Total']
        self.scat_gen = {'Suceptibles':None, 'Expuestos':None, 'Infectados':None, 'Recuperados':None}
        for i,val in enumerate(self.scat_gen.keys()):
            if i in [1,2]:
                self.scat_gen[val] = self.ax_gen.plot(intervalo, datos_gen[:,i], label= val)
        self.ax_gen.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        self.ax_gen.set_xlim(0,self.tamano_ventana)
        self.ax_gen.set_ylim(0,datos_gen.max())
        self.gen_vline = self.ax_gen.axvline(self.t)
        #print(self.gen_vline.get_xdata())
        self.ax_gen.set_title('Avance general')
        
        ##Particular
        datos_part = self.datos[self.region_part]#[0]
        self.scat_part = {'Suceptibles':None, 'Expuestos':None, 'Infectados':None, 'Recuperados':None}
        for j,val in enumerate(self.scat_part.keys()):
            if j in [0,1,2,3]:
                self.scat_part[val] = self.ax_part.plot(intervalo, datos_part[:,j], label= val)
        self.ax_part.legend(loc='upper center', ncol=4, fontsize = 'x-small')
        self.ax_part.set_xlim(0,self.tamano_ventana)
        self.ax_part.set_ylim(0,datos_part.max())
        self.part_vline = self.ax_part.axvline(self.t)
        self.ax_part.set_title('Avance del nodo:')
        
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
    
    def actualizar_nombre(self, Nombre):
        Nombre = Nombre.strip()
        if Nombre in self.nom_mun:
            self.region_part = Nombre.strip()
            self.update_part(self.t)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def auto_update(self, *args, **kwargs):
        print(args)
        print(kwargs)
        if not self.presionado:
            rango = range(self.t+1, self.t_final)
            self.button.label = 'Detener'
            self.presionado=True
            for i in rango:
                self.update(i)
                self.slider.set_val(i)
        elif self.presionado:
            self.presionado=False
            self.button.disconnect(self.button_cid)
    
    
    def update_gen(self, i):
        t_inicio = max(0,i-self.tamano_ventana//2)\
                    if i-self.tamano_ventana//2<self.t_final-self.tamano_ventana\
                    else self.t_final-self.tamano_ventana
        t_fin = max(self.tamano_ventana, self.tamano_ventana//2+i)\
                    if self.tamano_ventana//2+i<self.t_final else self.t_final
        self.gen_vline.set_xdata([i,i])
        #datos_gen = self.datos['Total'][:i]
        #for j, tipo in enumerate(self.edos_salud):
        #    self.scat_gen[tipo][0].set_data(np.arange(i),
        #                           datos_gen[:, j])
        self.ax_gen.set_xlim(t_inicio,t_fin)
    
    def update_part(self, i):
        t_inicio = max(0,i-self.tamano_ventana//2)\
                    if i-self.tamano_ventana//2<self.t_final-self.tamano_ventana\
                    else self.t_final-self.tamano_ventana
        t_fin = max(self.tamano_ventana, self.tamano_ventana//2+i)\
                    if self.tamano_ventana//2+i<self.t_final else self.t_final
        self.part_vline.set_xdata([i, i])
        #datos_part = self.datos[self.region_part][:i]
        #for j, tipo in enumerate(self.edos_salud):
        #    self.scat_part[tipo][0].set_data(np.arange(i),
        #                           datos_part[:, j])
        self.ax_part.set_xlim(t_inicio,t_fin)
        #self.ax_part.set_ylim(0,datos_part.max())

    def update_map(self, i):
        state = self.obtener_estado_mapa(i)
        self.mapa_scatter.set_offsets(state[:,:2])
        self.mapa_scatter.set_sizes(state[:,2])
        self.mapa_scatter.set_array(state[:,3])
        
        
    def obtener_estado_mapa(self, step):
        estado = np.zeros((len(self.nom_mun), 4)) #x, y, tamaño, color
        for j, region in enumerate(self.posiciones):
            coord = self.posiciones[region]
            datos = self.datos[region][step]
            infectados = datos[2]/datos.sum()
            estado[j] = np.array([coord[0], coord[1],
                                  self.tamanos[region], infectados])
        return(estado)

    def leer_datos(self,path):
        corrida = pd.read_pickle(path)
        with open('Datos/datos.pk', 'rb') as f:
            regiones = pk.load(f)
            self.posiciones = {k:norm_coord(regiones[k]['centro']) for k in regiones}
            self.tamanos = {k:max(np.log(regiones[k]['pob'])**2.5,20) for k in regiones}
        n_it = corrida.shape[0]
        self.datos=OrderedDict()
        self.datos['Total'] = corrida.iloc[:,:4].values
        for col in list(corrida.iloc[:,4:].columns):
            valores = np.zeros((n_it,4), dtype = np.uint)
            for i in range(n_it):
                valores[i] = np.array(corrida.loc[i, col])
            self.datos[col] = valores
        

def norm_coord(coord):
    coord = np.array(coord)[::-1]
    esq1 = np.array([-89.891994, 20.160681])
    esq2 = np.array([-88.986827, 21.288735])
    delta = esq2-esq1
    return (coord-esq1)/delta


if __name__=='__main__':
    vis = Visualizador('Pruebas/prueba1.pk')
