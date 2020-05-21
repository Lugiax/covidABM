#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mesa import Agent
from random import random, sample, gauss, choice, shuffle

#Algunas constantes
SUCEPTIBLE = 0
EXPUESTO = 1
INFECTADO = 2
RECUPERADO = 3
salud_to_str={0:'Suceptible', 1:'Expuesto', 2:'Infectado', 3:'Recuperado'}

class Individuo(Agent):
    
    def __init__(self, unique_id, model, edad, sexo):
        super().__init__(unique_id, model)
        self.mundo = self.model.mundo
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
            habitantes=len(self.mundo.nodes[self.casa_id]['habitantes'])
            moverse_entre_nodos = random() < (1/habitantes)
          else:
            moverse_entre_nodos = random() < 0.01

          if moverse_entre_nodos:
              if self.mundo.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.mundo.mover_en_nodos(self, 'aurrera')
              else:
                  self.mundo.mover_en_nodos(self, self.casa_id)
          else:
              self.mundo.siguiente_paso_aleatorio(self)

        elif self.salud == INFECTADO:

          moverse_entre_nodos = random() < 0.25

          if moverse_entre_nodos and not self.en_cuarentena==2:
              if self.mundo.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.mundo.mover_en_nodos(self, 'aurrera')
              else:
                  self.mundo.mover_en_nodos(self, self.casa_id)
          else:
              self.mundo.siguiente_paso_aleatorio(self)

        elif self.salud == RECUPERADO:

          moverse_entre_nodos = random() < 0.5

          if moverse_entre_nodos and not self.en_cuarentena==2:
              if self.mundo.nodes[self.nodo_actual]['tipo'] == 'casa':
                  self.mundo.mover_en_nodos(self, 'aurrera')
              else:
                  self.mundo.mover_en_nodos(self, self.casa_id)
          else:
              self.mundo.siguiente_paso_aleatorio(self)

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


class Individuo_base(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #Atributos del individuo
        self.mundo = model.mundo
        self.nodo_actual = None
        self.pos = None
        self.salud = model.SUCEPTIBLE
        self.sexo = None
        self.edad = None
        self.trabaja = True
        self.casa_id = None
        self.nodo_actual = None
        self.n_familiares = 0 #Número de familiares, incluyéndolo
        self.contador_interacciones = 0
        ##Atributos de comportamiento
        self.regresar_casa = False
        self.evitar_agentes = True
        self.evitar_sintomaticos = False
        self.activar_cuarentena = False ###Cambiar por: activar_cuarentena
        self.quedate_en_casa = False
        self.prob_movimiento = 0.005
        ##Atributos de la enfermedad
        ### Pasos para:
        self.pp_infectar = int(5*model.pp_dia)
        self.pp_infectar_var = 1*model.pp_dia
        self.pp_recuperar = int(9*model.pp_dia)
        self.pp_recuperar_var = 2*model.pp_dia
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
        if self.nodo_actual == self.casa_id and self.quedate_en_casa and\
        random()<1/self.n_familiares:
            self.mundo.siguiente_paso_aleatorio(self,
                                                 self.evitar_agentes)
            
        elif self.salud == self.model.INFECTADO and self.activar_cuarentena:
            if self.mundo.nodes[self.nodo_actual]['tipo']!='casa':
                self.mundo.mover_en_nodos(self, self.casa_id, pos = (0,0))
            else:
                #Se mueve evitando a los demás agentes
                self.mundo.siguiente_paso_aleatorio(self,
                                                     evitar_agentes = True)
                
        elif self.model.momento == 0: #Los que trabajan salen
            self.mundo.mover_en_nodos(self, 'ciudad')
            
        elif self.model.momento == 1: #Se mueven en su espacio
            self.mundo.siguiente_paso_aleatorio(self,
                                                 self.evitar_agentes)
        elif self.model.momento == 2: #Salen de trabajar
            if random() < self.prob_movimiento:
                if self.mundo.nodes[self.nodo_actual]['tipo'] == 'casa':
                    self.mundo.mover_en_nodos(self, 'ciudad')
                else:
                    self.mundo.mover_en_nodos(self, self.casa_id)
            else:
                self.mundo.siguiente_paso_aleatorio(self,
                                                     self.evitar_agentes)
                
        elif self.model.momento==3:
            if self.mundo.nodes[self.nodo_actual]['tipo'] != 'casa':
                self.mundo.mover_en_nodos(self, self.casa_id, pos = (0,0))
            
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
            contactos = self.mundo.obtener_contactos(self,
                                                      r=self.radio_de_infeccion)
            
            for a in contactos:
                self.contador_interacciones += 1
                if a.salud == self.model.SUCEPTIBLE and\
                random() < self.prob_contagiar*a.prob_infectarse:
                    a.salud = self.model.EXPUESTO

    def establecer_atributos(self, attrs):
        """
        Asigna los atributos que se encuentran en attrs al agente.
        Aunque no pertenezca nativamente al agente, este es agregado.
        """
        for at in attrs:
            try:
                getattr(self, at)
            except AttributeError:
                print(f'El atributo {at} no es nativo del agente')

            setattr(self, at, attrs[at])



#-----------------------------------------------------------------------------
class Individuo_2(Individuo_base):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.nodo_casa = None
        self.tray_nodos = list()

        ##Atributos de comportamiento
        self.evitar_sintomaticos = False
        self.distancia_paso = 1
        self.frac_mov_nodos = 0.1 #porcentaje de movimiento con respecto al movimiento total
   

    def step(self):

        dia = self.model.dia
        momento = self.model.n_paso%self.model.pp_dia
        fecha = (self.model.dia_cero + self.model.un_dia*dia).strftime('%Y-%m-%d')
        
        prob_mov = (1+self.model.movilidad.loc[fecha]/100)*self.prob_movimiento\
                if dia<self.model.movilidad.shape[0] else self.prob_movimiento

        if momento<2:
            disponibles = list(self.mundo.successors(self.nodo_actual))
            nuevo_nodo = choice(disponibles)
            prob_mov_nodos = self.mundo.obtener_peso(self.nodo_actual, nuevo_nodo)*prob_mov
            if random()<prob_mov_nodos:
                if nuevo_nodo == self.nodo_casa:
                    self.tray_nodos = []
                else:
                    self.tray_nodos.append(nuevo_nodo)
                self.mundo.mover_en_nodos(self, nuevo_nodo)

            elif random()<prob_mov:
                self.mundo.siguiente_paso_aleatorio(self, 
                                                evitar_agentes=self.evitar_agentes,
                                                evitar_sintomaticos=self.evitar_sintomaticos,
                                                radio = self.distancia_paso)

        elif momento>1:
            regresar = self.nodo_actual != self.nodo_casa
            if regresar:
                if len(self.tray_nodos)>0:
                    self.tray_nodos.pop()
                nuevo_nodo = self.tray_nodos[-1] if len(self.tray_nodos)>0\
                                                 else self.nodo_casa
                self.mundo.mover_en_nodos(self, nuevo_nodo)
            else:
                self.mundo.siguiente_paso_aleatorio(self, 
                                                evitar_agentes=self.evitar_agentes,
                                                evitar_sintomaticos=self.evitar_sintomaticos,
                                                radio = self.distancia_paso)


        """
        if self.unique_id == 0: print(
            f'Día: {dia}\nMomento: {momento}'
            )

        
        self.regresar_casa = momento>1 and self.nodo_actual != self.nodo_casa
        print(f'Regresar a casa {self.regresar_casa}')
        if self.regresar_casa or (momento<2 and random()<self.prob_mov_nodos):
            if self.regresar_casa:
                print(f'\tPara seleccionar {self.tray_nodos}')
                nuevo_nodo = self.tray_nodos.pop()\
                            if len(self.tray_nodos)>0 else self.nodo_casa

                print(
                    f'\tSe regresa a casa, va a {nuevo_nodo}. Restante {self.tray_nodos}'
                    )
            else:
                self.regresar_casa = False
                disponibles = list(self.mundo.successors(self.nodo_actual))
                nuevo_nodo = choice(disponibles)
                print(
                    f'\tPosible movimiento entre nodos de {self.nodo_actual} a {nuevo_nodo}'
                    )

            if (self.regresar_casa or 
            random()<self.mundo.obtener_peso(self.nodo_actual, nuevo_nodo)):
                if nuevo_nodo == self.nodo_casa:
                    self.tray_nodos = []
                else:
                    self.tray_nodos.append(nuevo_nodo)

                print(f'\t\tSe moverá al nodo {nuevo_nodo} (debe coincidir con el anterior)')
                self.mundo.mover_en_nodos(self, nuevo_nodo)
                ## Regresa a casa, se resetea el vector de trayectoria
                print(
                    f'\t\tNueva trayectoria: {self.tray_nodos}'
                    )

        else:
            prob_mov = (1+self.model.movilidad[dia,1]/100)*self.prob_movimiento\
                if dia<len(self.model.movilidad) else self.prob_movimiento
            if self.unique_id == 0: print(
                    f'\tSe mueve en su espacio, con probabilidad {prob_mov}'
                    )
            
            if random()<prob_mov:
                self.mundo.siguiente_paso_aleatorio(self, 
                                                evitar_agentes=self.evitar_agentes,
                                                evitar_sintomaticos=self.evitar_sintomaticos,
                                                radio = self.distancia_paso)
                if self.unique_id == 0: print(
                    f'\t\tSe movió'
                    )
                
        """
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
        
        

    def aplicar_medidas(self, medidas = {}):
        self.establecer_atributos(medidas)