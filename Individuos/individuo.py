#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mesa import Agent
from random import random, sample, gauss

#Algunas constantes
SUCEPTIBLE = 0
EXPUESTO = 1
INFECTADO = 2
RECUPERADO = 3
salud_to_str={0:'Suceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}

class Individuo(Agent):
    
    def __init__(self, unique_id, model, edad, sexo):
        super().__init__(unique_id, model)
        self.ciudad = self.model.ciudad
        self.pos = None
        self.salud = SUCEPTIBLE
        self.sexo = sexo
        self.edad = edad
        self.pasos_infectado=0
        self.casa_id = None
        self.nodo_actual = None
        self.R0 = 6 # Índice de reproducción de la enfermedad entre individuos.
        self.pasos_para_infectar = 15
        self.pasos_para_recuperarse = 15
        self.asintomatico = False  # transmite, sin sintomas, no esta en cuarentena y su movilidad es regular. pend.
        self.en_cuarentena = 0 # 1: reduce movilidad, 2: aislamiento total.
        self.diagnosticado = False  # Se sabe que tiene la enfermedad y debe de estar en cuarentena. pend.
        self.politica_encasa = False
        self.politica_higiene = 0.0 # porcentaje de reduccion de la probalidad de contagio.
        self.mortalidad_apriori = 0.05 # pend.

    def step(self):
        if self.salud == SUCEPTIBLE or self.salud == EXPUESTO:

          if self.politica_encasa:
            habitantes=len(self.ciudad.nodes[self.casa_id]['habitantes'])
            moverse_entre_nodos = random() < (1/habitantes)
          else:
            moverse_entre_nodos = random() < 0.01

          if moverse_entre_nodos:
              if self.ciudad.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.ciudad.mover_en_nodos(self, 'aurrera')
              else:
                  self.ciudad.mover_en_nodos(self, self.casa_id)
          else:
              self.ciudad.siguiente_paso_aleatorio(self)

        elif self.salud == INFECTADO:

          moverse_entre_nodos = random() < 0.25

          if moverse_entre_nodos and not self.en_cuarentena==2:
              if self.ciudad.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.ciudad.mover_en_nodos(self, 'aurrera')
              else:
                  self.ciudad.mover_en_nodos(self, self.casa_id)
          else:
              self.ciudad.siguiente_paso_aleatorio(self)

        elif self.salud == RECUPERADO:

          moverse_entre_nodos = random() < 0.5

          if moverse_entre_nodos and not self.en_cuarentena==2:
              if self.ciudad.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.ciudad.mover_en_nodos(self, 'aurrera')
              else:
                  self.ciudad.mover_en_nodos(self, self.casa_id)
          else:
              self.ciudad.siguiente_paso_aleatorio(self)

        self.interactuar()
        
        ## Se revisa la evolución de su salud
        if self.salud == EXPUESTO:
            self.pasos_infectado += 1
            if self.pasos_infectado > self.pasos_para_infectar:
                self.salud = INFECTADO
        elif self.salud == INFECTADO:
            self.pasos_infectado += 1
            if self.pasos_infectado>self.pasos_para_infectar + self.pasos_para_recuperarse:
                self.salud = RECUPERADO
    
    def interactuar(self):
        ## Se selecciona un número de agentes por contagiar de entre los que
        ## se encuentran en su mismo nodo, solamente si está infectado
        x, y = self.pos
        contactos = self.model.ciudad.nodes[self.nodo_actual]['espacio'][x][y]
        por_contagiar = self.R0//2
        prob_contagio = .8*(1-self.politica_higiene)
        if self.salud == INFECTADO and not self.en_cuarentena == 2:
            for a in sample(contactos, min(por_contagiar, len(contactos))):
                if a.salud == SUCEPTIBLE and random() < prob_contagio:
                    a.salud = EXPUESTO

##----------------------------------------------------------------------------


class Individuo_basico(Agent):
    
    def __init__(self, unique_id, model, edad, sexo):
        super().__init__(unique_id, model)
        self.ciudad = self.model.ciudad
        self.pos = None
        self.salud = SUCEPTIBLE
        self.sexo = sexo
        self.edad = edad
        self.pasos_infectado=0
        self.casa_id = None
        self.nodo_actual = None
        self.R0 = 10
        self.pasos_para_infectar = 40
        self.pasos_para_recuperarse = 30

    def step(self):
        moverse_entre_nodos = random() < 0.005
        if moverse_entre_nodos:
            if self.ciudad.nodes[self.nodo_actual]['tipo'] == 'casa':
                self.ciudad.mover_en_nodos(self, 'aurrera')
            else:
                self.ciudad.mover_en_nodos(self, self.casa_id)
        else:
            #if self.nodo_actual=='aurrera':
            #    print(f'Ind {self.unique_id} da paso aleatorio')
            self.ciudad.siguiente_paso_aleatorio(self,
                                                 evitar_agentes=True)
            
        self.interactuar()
        
        ## Se revisa la evolución de su salud
        if self.salud == EXPUESTO:
            self.pasos_infectado += 1
            if self.pasos_infectado > self.pasos_para_infectar:
                self.salud = INFECTADO
        elif self.salud == INFECTADO:
            self.pasos_infectado += 1
            if self.pasos_infectado>self.pasos_para_infectar + self.pasos_para_recuperarse:
                self.salud = RECUPERADO
    
    def interactuar(self):
        ## Se selecciona un número de agentes por contagiar de entre los que
        ## se encuentran en su mismo nodo, solamente si está infectado

        if self.salud == INFECTADO:
            x, y = self.pos
            contactos = self.ciudad.obtener_contactos(self, r = 1)
            por_contagiar = self.R0
            prob_contagio = .8
            for a in sample(contactos, min(por_contagiar, len(contactos))):
                if a.salud == SUCEPTIBLE and random() < prob_contagio:
                    a.salud = EXPUESTO



#-----------------------------------------------------------------------------
class Individuo_2(Agent):
    
    def __init__(self, unique_id, model, edad, sexo):
        super().__init__(unique_id, model)
        #Atributos del individuo
        self.ciudad = model.ciudad
        self.pos = None
        self.salud = model.SUCEPTIBLE
        self.sexo = sexo
        self.edad = edad
        self.trabaja = True
        self.casa_id = None
        self.nodo_actual = None
        self.n_familiares = 0 #Número de familiares, incluyéndolo
        ##Atributos de comportamiento
        self.evitar_agentes = True
        self.en_cuarentena = False
        self.quedate_en_casa = False
        self.prob_movimiento = 0.005
        ##Atributos de la enfermedad
        ### Pasos para:
        self.pp_infectar = 14*model.pp_dia
        self.pp_infectar_var = 4*model.pp_dia
        self.pp_recuperar = 10*model.pp_dia
        self.pp_recuperar_var = 3*model.pp_dia
        ##Atributos ante la enfermedad
        self.prob_contagiar = 0.5
        self.prob_infectarse = 0.8
        self.prob_recuperarse = 0.8 #Sino muere
        self.radio_de_infeccion = 0
        self.asintomatico = False
        ###Estas variables contarán los pasos que faltan para:
        self.pp_infectarse = int(gauss(self.pp_infectar,
                                       self.pp_infectar_var/2))
        self.pp_recuperarse = int(gauss(self.pp_recuperar,
                                        self.pp_recuperar_var/2))
        
        

    def step(self):
        if self.quedate_en_casa and random()<1/self.n_familiares:
            self.ciudad.siguiente_paso_aleatorio(self,
                                                 self.evitar_agentes)
            
        elif self.salud == self.model.INFECTADO and self.en_cuarentena:
            if self.ciudad.nodes[self.nodo_actual]['tipo']!='casa':
                self.ciudad.mover_en_nodos(self, self.casa_id, pos = (0,0))
            else:
                #Se mueve evitando a los demás agentes
                self.ciudad.siguiente_paso_aleatorio(self,
                                                     evitar_agentes = True)
                
        elif self.model.momento == 0: #Los que trabajan salen
            self.ciudad.mover_en_nodos(self, 'ciudad')
            
        elif self.model.momento == 1: #Se mueven en su espacio
            self.ciudad.siguiente_paso_aleatorio(self,
                                                 self.evitar_agentes)
        elif self.model.momento == 2: #Salen de trabajar
            if random() < self.prob_movimiento:
                if self.ciudad.nodes[self.nodo_actual]['tipo'] == 'casa':
                    self.ciudad.mover_en_nodos(self, 'ciudad')
                else:
                    self.ciudad.mover_en_nodos(self, self.casa_id)
            else:
                self.ciudad.siguiente_paso_aleatorio(self,
                                                     self.evitar_agentes)
                
        elif self.model.momento==3:
            if self.ciudad.nodes[self.nodo_actual]['tipo'] != 'casa':
                self.ciudad.mover_en_nodos(self, self.casa_id, pos = (0,0))
            
        self.interactuar()
        
        ## Se revisa la evolución de su salud
        if self.salud == self.model.EXPUESTO:
            self.pp_infectarse -= 1
            if self.pp_infectarse == 0:
                self.salud = self.model.INFECTADO
        elif self.salud == self.model.INFECTADO:
            self.pp_recuperarse -= 1
            if self.pp_recuperarse == 0:
                self.salud = self.model.RECUPERADO
    
    def interactuar(self):
        ## Se selecciona un número de agentes por contagiar de entre los que
        ## se encuentran en su mismo nodo, solamente si está infectado

        if self.salud == self.model.INFECTADO:
            x, y = self.pos
            contactos = self.ciudad.obtener_contactos(self,
                                                      r=self.radio_de_infeccion)
            
            for a in contactos:
                if a.salud == self.model.SUCEPTIBLE and\
                random() < self.prob_contagiar*a.prob_infectarse:
                    a.salud = self.model.EXPUESTO