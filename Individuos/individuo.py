#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mesa import Agent
from scipy.spatial.distance import euclidean, cityblock, chebyshev
from math import exp

class Individuo(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #Atributos del individuo
        self.mundo = model.mundo
        self.pos = None
        self.salud = model.SUSCEPTIBLE
        self.sexo = None
        self.edad = None
        self.trabaja = True
        self.casa_id = None
        self.nodo_actual = None
        self.nodo_casa = None
        self.n_familiares = 0 #Número de familiares, incluyéndolo
        self.contador_interacciones = 0
        ##Atributos de comportamiento
        self.regresar_casa = False
        self.evitar_agentes = False
        self.evitar_sintomaticos = False
        self.activar_cuarentena = False ###Cambiar por: activar_cuarentena
        self.quedate_en_casa = False
        self.prob_movimiento = 0.005
        self.distancia_paso = 1
        self.alpha = 0.1
        ##Atributos de la enfermedad
        ### Pasos para:
        self.dp_infectar = 7
        self.dp_infectar_var = 1
        self.dp_recuperar = 9
        self.dp_recuperar_var = 2
        ##Atributos ante la enfermedad
        self.prob_contagiar = 0.5
        self.prob_infectarse = 0.8
        self.prob_recuperarse = 0.8 
        self.radio_de_infeccion = 0
        self.asintomatico = False
        ###Estas variables contarán los pasos que faltan para:
        self.pp_infectarse = int(model.rand.gauss(self.dp_infectar,
                                       self.dp_infectar_var/2))*model.pp_dia
        self.pp_recuperarse = int(model.rand.gauss(self.dp_recuperar,
                                        self.dp_recuperar_var/2))*model.pp_dia
        
        

    def step(self):
        if self.model.rand.random() < self.prob_movimiento:
            self.mundo.siguiente_paso_aleatorio(self, 
                                            evitar_agentes=self.evitar_agentes,
                                            evitar_sintomaticos=self.evitar_sintomaticos,
                                            radio = self.distancia_paso)

        self.interactuar()

        self.revisar_salud()
    
    def interactuar(self):
        ## Se selecciona un número de agentes por contagiar de entre los que
        ## se encuentran en su mismo nodo, solamente si está infectado
        if self.salud == self.model.INFECTADO:
            x, y = self.pos
            contactos = self.mundo.obtener_contactos(self,
                                                      r=self.radio_de_infeccion)
            
            for a in contactos:
                self.contador_interacciones += 1
                distancia = chebyshev(self.pos, a.pos)
                probabilidad_infectarse = exp(-self.alpha*distancia)*self.prob_contagiar
                if a.salud == self.model.SUSCEPTIBLE and\
                self.model.rand.random() < probabilidad_infectarse:
                    a.salud = self.model.EXPUESTO

    def revisar_salud(self):
        ## Se revisa la evolución de su salud
        if self.salud == self.model.EXPUESTO:
            self.pp_infectarse -= 1
            infectarse = self.pp_infectarse == 0 and self.model.rand.random()<self.prob_infectarse
            if infectarse:
                self.salud = self.model.INFECTADO
                if self.activar_cuarentena and self.nodo_actual!='cuarentena':
                    self.mundo.mover_en_nodos(self, 'cuarentena', (0,0))

            elif self.pp_infectarse == 0:
                self.salud = self.model.SUSCEPTIBLE
                self.pp_infectarse = int(self.model.rand.gauss(self.dp_infectar,
                                       self.dp_infectar_var/2))*self.model.pp_dia

        elif self.salud == self.model.INFECTADO:
            self.pp_recuperarse -= 1
            if self.pp_recuperarse == 0:
                self.salud = self.model.RECUPERADO
                if self.activar_cuarentena:
                    self.mundo.mover_en_nodos(self, self.nodo_casa)



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
            if at == 'dp_infectar':
                self.pp_infectarse = int(self.model.rand.gauss(self.dp_infectar,
                                         self.dp_infectar_var/2))*self.model.pp_dia
            elif at == 'dp_recuperar':
                self.pp_recuperarse = int(self.model.rand.gauss(self.dp_recuperar,
                                         self.dp_recuperar_var/2))*self.model.pp_dia


    def aplicar_medidas(self, medidas = {}):
        self.establecer_atributos(medidas)


#-----------------------------------------------------------------------------
class Individuo_2(Individuo):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.nodo_casa = None
        self.tray_nodos = list()

        ##Atributos de comportamiento
        self.evitar_sintomaticos = False
        self.distancia_paso = 1
        self.frac_mov_nodos = 0 #porcentaje de movimiento con respecto al movimiento total
   

    def step(self):

        if self.nodo_actual=='cuarentena':
            self.revisar_salud()
            return

        dia = self.model.dia
        momento = self.model.n_paso%self.model.pp_dia
        fecha = (self.model.dia_cero + self.model.un_dia*dia).strftime('%Y-%m-%d')
        
        try:
            if self.model.params['tomar_movilidad']:
                prob_mov= (1+self.model.movilidad.loc[fecha]/100)*self.prob_movimiento
            else:
                prob_mov = self.prob_movimiento
        except:
            prob_mov= self.prob_movimiento


        if momento<2:
            disponibles = list(self.mundo.successors(self.nodo_actual))
            nuevo_nodo = self.model.rand.choice(disponibles)
            prob_mov_nodos = self.mundo.obtener_peso(self.nodo_actual, nuevo_nodo)*prob_mov
            if self.model.rand.random()<prob_mov:
                if self.model.rand.random()<prob_mov_nodos:
                    if nuevo_nodo == self.nodo_casa:
                        self.tray_nodos = []
                    else:
                        self.tray_nodos.append(nuevo_nodo)
                    self.mundo.mover_en_nodos(self, nuevo_nodo)
                else:
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

        self.interactuar()
        self.revisar_salud()

if __name__=='__main__':
    p1 = (1,1)
    p2 = (2,2)
    dist = chebyshev(p1,p2)#euclidean(p1,p2) 
    print(dist)
    print(exp(-0.5*dist))