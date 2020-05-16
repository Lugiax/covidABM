#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:47:42 2020

@author: carlos
"""

from networkx import MultiDiGraph
import networkx as nx
from mesa import Agent, Model
from mesa.space import MultiGrid
from random import gauss, random, sample, choice, choices, randrange, shuffle
import numpy as np
import matplotlib.pyplot as plt

class Mundo(MultiDiGraph):
    """
    La ciudad es la que se encargará de generar los nodos que representen
    lugares de la ciudad como casas, centros comerciales, iglesias, o lo que sea.
    Tiene 4 métodos:
        * Generar los individuos
        * Crear los hogares
        * Hacer las conexiones entre los nodos casas y los lugares
        * Mover los individuos a nodos específicos
    La ciudad está representada como un grafo.
    """
    def __init__(self, model, agent_object, verbose = False):
        super().__init__()
        self.verbose = verbose
        self.agent_object = agent_object
        self.model = model
        self.p_asintomatico = 0
        self.p_matrimonio = 0.7
        self.promedio_hijos = 2
        self.porcentaje_hombres = 0.49
        self.casasids = []
        self.lugaresids = []
        self.agentes_a_asignar=[]
    
    def generar_individuos(self, N, ids = [], attrs={}):
        """
        Considerando el porcentaje de hombres y mujeres en la población, y la
        distribución de las edades en la población (considerando una distribución
        normal) se procede a generar la población.
        """
        if len(ids)!=N:
            ids = [i for i in range(N)]
        agentes = []
        for i in range(N):
            agente = self.agent_object(ids[i], self.model)

            attrs['sexo'] = 'h' if random()<=self.porcentaje_hombres else 'm'

            edad = gauss(25,20)
            if edad < 0:
                attrs['edad'] = 0
            elif edad > 90:
                attrs['edad'] = 90

            if random()<self.p_asintomatico:
                attrs['asintomatico'] = True
            else: 
                attrs['asintomatico'] = False
            ##Se establecen los atributos en attrs
            agente.establecer_atributos(attrs)
            #for at in attrs:
            #    setattr(agente, at, attrs[at])
            
            agentes.append(agente)
        return agentes
    
    def crear_hogares(self):
        """
        Consideranco la probabilidad de que en una casa exista un matrimonio, 
        y el promedio de hijos menores a 23 años en cada casa, se asignan
        individuos a cada nodo hogar.
        Se consideran personas solteras con y sin hijos.
        """
        #Se generan las listas con cada tipo de individuos a repartir
        hombres = [ind for ind in self.agentes_a_asignar if ind.sexo == 'h' and ind.edad>=23]
        mujeres = [ind for ind in self.agentes_a_asignar if ind.sexo == 'm' and ind.edad>=23]
        hijos = [ind for ind in self.agentes_a_asignar if ind.edad < 23]
        if self.verbose: print(f'Cantidad de hombres {len(hombres)}, mujeres {len(mujeres)}, hijos {len(hijos)}')
        contador_nodos = 0
        while len(hombres)>0 or len(mujeres)>0:
            a_agregar = []
            matrimonio = random() < self.p_matrimonio
            ##Se asignan primero los jefes de familia (hombre y/o mujer)
            if len(hombres)*len(mujeres)>0 and matrimonio:
                a_agregar.append(hombres.pop())
                a_agregar.append(mujeres.pop())
            else:
                seleccion = sample([hombres, mujeres], 2)
                if len(seleccion[0])>0:
                    a_agregar.append(seleccion[0].pop())
                else:
                    a_agregar.append(seleccion[1].pop())
            
            ##Se asignan los hijos a cada casa
            n_hijos = abs(int(gauss(self.promedio_hijos, 0.5)))
            while len(hijos)>0 and n_hijos>0:
                a_agregar.append(hijos.pop())
                n_hijos-=1
            
            ##Se procede a crear el nodo correspondiente
            self.crear_nodo(contador_nodos, tipo = 'casa',
                            ocupantes = a_agregar)
            
            if self.verbose: print(f'En la casa {contador_nodos} hay {len(a_agregar)} personas. Matrimonio : {matrimonio}')
                
            self.casasids.append(contador_nodos)
            contador_nodos += 1
        
        ##Si sobran hijos, entonces se agregan al azar a todos los nodos
        while len(hijos) > 0:
            hijo = hijos.pop()
            while True:
                idcasa = choice(list(self.nodes))
                espacio_casa = self.nodes[idcasa]['espacio']
                if len(espacio_casa.empties) == 0 :
                    continue
                
                if self.verbose: print(f'Se agrega un hijo a la casa {idcasa}... ', end='')
                self.nodes[idcasa]['habitantes'].append(hijo.unique_id)
                #print('Espacios en casa', espacio_casa.empties)
                
                espacio_casa.place_agent(hijo, espacio_casa.empties[-1])
                hijo.casa_id = idcasa
                hijo.nodo_actual = idcasa
                if self.verbose: print('Agregado')
                break
        
    def conectar_a_casas(self, nid):
        for i in list(self.nodes):
            if self.nodes[i]['tipo']=='casa':
                self.conectar_nodos(nid,i)
                
    def conectar_nodos(self, nodo1, nodo2):
        self.add_edge(nodo1, nodo2,
                      peso = 1)
            
    def crear_nodo(self,nodo_id, tipo, ocupantes = [], tamano = None,
                   ind_pos_def = None):
        if tipo == 'municipio':
            assert len(ocupantes)>0, 'No hay ocupantes a asignar en el munucupio'
            if not tamano:
                tamano = 2#len(ocupantes)//2+1
            habitantes = [ind.unique_id for ind in ocupantes]
        
        else:
            if not tamano:
                tamano = 20
            habitantes = []
        
        espacio = MultiGrid(width = tamano,
                            height = tamano,
                            torus = False)
        
        #Cómo colocar a los individuos
        if ind_pos_def=='vacios': 
            disponibles = espacio.empties[::]
            shuffle(disponibles)
            
        
        for i in ocupantes:
            i.nodo_casa = nodo_id if tipo=='municipio' else None
            i.n_familiares = len(habitantes) if tipo=='casa' else 0
            i.nodo_actual = nodo_id
            if isinstance(ind_pos_def, (list, tuple)):
                #print('Posición definida')
                i_pos = ind_pos_def
            elif ind_pos_def=='vacios':
                #print('Se pone en los vacíos')
                i_pos = disponibles.pop()
            elif ind_pos_def=='aleatorio':
                #print('Aleatorios')
                i_pos = (randrange(tamano), randrange(tamano))
            espacio.place_agent(i, i_pos)
        
        
        self.add_node(nodo_id, tipo = tipo,
                      habitantes = habitantes,
                      espacio = espacio)
        
    def mover_en_espacio(self, ind, nueva_pos):
        """
        Mueve al individuo en el espacio en el que se encuentra a una
        nueva posición
        """
        espacio = self.nodes[ind.nodo_actual]['espacio']
        espacio.move_agent(ind, nueva_pos)
    
    
    def mover_en_nodos(self, ind, nuevo_nodo_id, pos = None):
        """
        Mueve a un individuo del nodo en el que se encuentra al nuevo_nodo_id.
        En caso de que no se indique la posición, se colocará al individuo en
        una casilla aleatoria de todo el espacio.
        """
        self.nodes[ind.nodo_actual]['espacio'].remove_agent(ind)
        nuevo_espacio= self.nodes[nuevo_nodo_id]['espacio']
        if not pos:
            pos = (randrange(nuevo_espacio.width),
                   randrange(nuevo_espacio.height))
        nuevo_espacio.place_agent(ind, pos)
        ind.nodo_actual = nuevo_nodo_id
    
    def contactos(self, ind):
        """
        Devuelve un iterable con todos los individuos que se encuentran en la
        celda donde está ind, incluyéndolo. NO UTILIZAR
        """
        x, y = ind.pos
        return self.nodes[ind.nodo_actual]['espacio'][x][y]
    
    def obtener_contactos(self, ind, r=0):
        """
        Devuelve una lista con todos los agentes vecinos en un radio r,
        excluyendo al agente en cuestión.
        """
        espacio = self.obtener_espacio(ind)
        vecinos = espacio.get_neighbors(pos = ind.pos,
                                        moore = True,
                                        include_center = True,
                                        radius = r)
        vecinos.remove(ind)
        return vecinos
    
    def obtener_agentes_en_nodo(self, nodo_id):
        espacio = self.obtener_espacio(nodo_id)
        agentes = []
        for elementos, x, y in espacio.coord_iter():
            for agente in elementos:
                agentes.append(agente)
        return agentes
    
    def obtener_peso(self, u, v, arista=0):
        return self.get_edge_data(u,v, key=arista)['peso']
        
        
    def obtener_familia(self, ind):
        """
        Devuelve una lista con todos los id de cada elemento de la familia,
        incluyendo a ind
        """
        return self.nodes[ind.casa_id]['habitantes']
    

    def siguiente_paso_aleatorio(self, ind, evitar_agentes = False,
                                    evitar_sintomaticos = False, radio = 1):
        """
        Mueve al individuo de forma aleatoria en el espacio en el que 
        se encuentra. En caso de que evitar_agentes == True, se selecciona
        la siguiente posición del espacio de Moore disponible.
        """
        x, y = ind.pos
        espacio = self.obtener_espacio(ind)
        
        if not evitar_agentes and not evitar_sintomaticos:
            paso_x, paso_y = choices(list(range(-radio,radio+1)), k=2)
            nueva_x = x + paso_x
            nueva_y = y + paso_y
            nueva_x, nueva_y = self.ajustar_posicion((nueva_x, nueva_y),
                                                     espacio)

        else:
            vecindario = espacio.get_neighborhood(pos = (x,y),
                                                  moore = True,
                                                  include_center = True,
                                                  radius = radio)
            shuffle(vecindario)
            nueva_x, nueva_y = x, y

            if evitar_agentes:
                disponibles = [pos for pos in vecindario\
                               if espacio.is_cell_empty(pos)]
                if len(disponibles)!=0:
                    nueva_x, nueva_y = disponibles.pop()

            elif evitar_sintomaticos:
                for pos in vecindario:
                    for i in espacio.iter_cell_list_contents(pos):
                        if i.asintomatico == True or i is self:
                            continue
                    nueva_x, nueva_y = pos
                    break

        espacio.move_agent(ind, (nueva_x, nueva_y))
        
    def ajustar_posicion(self, pos, espacio):
        """
        Recibe la posicón y la instancia del espacio a donde se desea ajustar
        """
        x, y = pos
        x = max(x, 0)
        y = max(y, 0)
        x = min(x, espacio.width-1)
        y = min(y, espacio.height-1)
        return(x,y)

    def obtener_espacio(self, nodo_id_o_individuo):
        """
        Devuelve el espacio al que pertenece un individuo, o que contiene
        un nodo
        """
        if isinstance(nodo_id_o_individuo, Agent):
            return self.nodes[nodo_id_o_individuo.nodo_actual]['espacio']
        else:
            return self.nodes[nodo_id_o_individuo]['espacio']
        

    def ver_espacio(self, nodo_id, figsize = (5,5)):
        """
        Se asume que el espacio es un arreglo MultiGrid, donde cada elemento
        es un conjunto que puede contener más de un individuo
        """
        espacio = self.obtener_espacio(nodo_id)
        ##Primero se cuentan los individuos
        cuenta = np.zeros((espacio.height, espacio.width),
                          dtype = np.uint)
        total = 0
        for i, x in enumerate(espacio.grid):
            for j, y in enumerate(x):
                n_individuos_en_celda = len(y)
                cuenta[i][j] = n_individuos_en_celda
                total += n_individuos_en_celda
        
        ##Luego se muestra una representación
        plt.figure(figsize = figsize)
        plt.imshow(cuenta, vmin = 0, vmax = cuenta.max())
        plt.colorbar()
        plt.show()
    
    def visualizar(self, pos = None, with_labels = False, figsize =(7,7)):
        fig, ax = plt.subplots(figsize = figsize)
        nx.draw(self, pos = pos, with_labels = with_labels, ax = ax)
        plt.show(block=True)
        
if __name__=='__main__':
    import networkx as nx
    modelo = Model()
    modelo.num_ind = 10
    class Individuo(Agent):
        def __init__(self, unique_id, model, edad, sexo):
            super().__init__(unique_id, model)
            self.model = model
            self.edad = edad
            self.sexo = sexo
    
    ciudad = Mundo(modelo, Individuo)
    ciudad.generarindividuos()
    #Se crean las casas distribuyendo los individuos
    ciudad.crear_hogares()
    
    #Se agrega una tienda a la ciudad y se conecta con todas las casas
    ciudad.crear_nodo('aurrera', tipo='tienda', tamano=10)
    ciudad.conectar_a_casas('aurrera')
    
    #nx.draw(ciudad)
    #ciudad.ver_espacio(1)
    aurrera = ciudad.obtener_espacio('aurrera')
    print(aurrera.get_neighborhood((8,8),True))
    ind1 = ciudad.agentes_a_asignar[0]
    vecinos_ind1 = ciudad.obtener_contactos(ind1, r = 0)
    print(ind1 in vecinos_ind1)