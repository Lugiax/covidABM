# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:25:04 2015
@author: carlos
Algorítmo Genético con variables múltiples
"""

import time
from math import log
import random as rnd
from binstr import b_bin_to_gray, b_gray_to_bin
try:
    from multiprocessing import Pool
except:
    print('Sin la capacidad de usar paralelización. Instalar módulo de ParallelPython')

class AG(object):
    
    def __init__(self,deb=False,seed=None):
        
        self.deb=deb ## Para depurar el programa  
        if seed:
            rnd.seed(seed)
        
        self.dxmax=0.005 # Presición
        self.Nind=50 # Número de individuos por default
        self.Ngen=100 # Número de generaciones por default
        self.prop_cruz=0.3 # Proporción de cruzamiento entre parejas por default
        self.prob_mut=0.05 # Probabilidad de mutación de un individuo por default
        self.elit=1 # 1-Con elitismo ; 2-Sin elitismo
        self.resultados_fitness=list()        
        
        self.hist_mej=[[],[],[]] ## Variable para guardar la historia del mejor individuo [Generación, Fitness, x]
            
        if deb:print('\nSe ha iniciado con el algoritmo, favor de introducir los parámetros')
            
        
        ## Funciones de codificación y decodificación
        self.decod= lambda n, a, b: (int(b_gray_to_bin(n),2)*int((b-a)/self.dxmax)/self.dmax+int(a/self.dxmax))*self.dxmax
        self.cod= lambda n: b_bin_to_gray(("{:0>"+str(self.l)+"}").format(bin(n).lstrip("0b")))

    def decodificado(self, genoma):
    	dec=list()
    	for i in range(self.nvars):
    		if self.comun:
    			a=self.a
    			b=self.b
    		else:
    			a=self.vars[i][1]
    			b=self.vars[i][2]
    		#print(genoma[i*self.l:(i+1)*self.l])
    		#print(int(genoma[i*self.l:(i+1)*self.l],2),int((self.b_a)/self.dxmax)/self.dmax,int(a/self.dxmax))
    		dec.append(self.decod(genoma[i*self.l:(i+1)*self.l],a,b))
    	return(dec)
        
        
    def parametros(self, pres=None, Nind=None, Ngen=None, prop_cruz=None, prob_mut=None, elit=1, optim=0, tipo_cruz='2p', pruebas=1, procesos=None):
        if pres:
            self.dxmax=pres
        if Nind:
            self.Nind=Nind+Nind%2
        if Ngen:
            self.Ngen=Ngen
        if prop_cruz:
            self.prop_cruz=prop_cruz
        if prob_mut:
            self.prob_mut=prob_mut
        if optim==1:
            self.max=True
            if self.deb:print('Configurado para buscar el valor máximo')
        else:
            self.max=False
            if self.deb:print('Configurado para buscar el valor mínimo')
        
        

        self.cores=procesos
        if procesos:
            self.pool = Pool(processes = procesos)
        
        self.tipo_cruz=tipo_cruz
        self.elit=elit
        self.pruebas=pruebas
        
    def Fobj(self, f, datos=None): ## Introducir la función a evaluar
        self.f_obj=f
        self.datos=datos
        
        if self.deb:print('Se ha introducido correctamente la función objetivo')
    
    def variables(self, variables=None, comun=None):
    	##Si se ingresa la variable comun, deberá tener la siguiente forma:
    	##  comun=[#deVariables, lim_inferior, lim_superior]
    	if comun:
    		dabmax=comun[2]-comun[1]
    		self.nvars=comun[0]
    		self.comun=True
    		self.a=comun[1]
    		self.b=comun[2]
    	elif variables:
    		dabmax=0
    		self.vars=variables
    		for var in variables:
    		    dab=var[2]-var[1]
    		    if dab>dabmax: dabmax=dab
    		self.nvars=len(variables)
    		self.comun=False
         
    	self.b_a=dabmax	
    	self.l=int(log(((dabmax)/self.dxmax)+1,2))+1
    	self.dmax=2**self.l-1
    	if self.deb==True:print('Ingreso de variables exitoso. ValMax={}, LongitudCadena={}'.format(dabmax,self.l))
            
            

    def fitnes(self, pob):
        
        if not self.cores:
            resultados_fitness=list()
            for ind in pob:
                ##Se utiliza la función objetivo para calcular el fitness
                #print('Decod ind',self.decodificado(ind))
                fit_ind=self.f_obj(*self.decodificado(ind))
                resultados_fitness.append([fit_ind,ind])
        else:
            #salidas = Queue()
            resultados_fitness=[[self.pool.apply(self.f_obj,
                                                args=self.decodificado(ind)),
                                ind] for ind in pob]

        resultados_fitness.sort(reverse=self.max)
        return(resultados_fitness)

    def crearPob(self,N_ind=None, d_max=None):
        pob=[]
        
        if not N_ind:
            N_ind=self.Nind
        if not d_max:
            d_max=self.dmax
            
        for i in range(N_ind):
            genoma=''
            for j in range(self.nvars):
                rand=rnd.randrange(0,d_max)
                genoma+=self.cod(rand)
            #print('Lgenoma',len(genoma))
            pob.append(genoma)
        
        return(pob)

    def seleccion(self,fit):
    ## La selección se hará de tipo Vasconcelos
        cruza=list()
        #print('Cruzamiento Nind:',int(self.Nind/2))
        for i in range(int(self.Nind/2)):
            #print '\nSeleccion',i+1
            ind1=fit[i][1]
            ind2=fit[self.Nind-i-1][1]
            cruza.append([ind1,ind2])
        return cruza
        
    def cruzamiento(self,tipo,pob):
        pob1=list()
        for pareja in pob:
            ind1_0,ind2_0=pareja
            ind1=''
            ind2=''
            if tipo=='uniforme':
                for bit in range(len(ind1_0)):
                    nbit1=ind1_0[bit]
                    nbit2=ind2_0[bit]
                    dados=rnd.random()
                    #print (dados)
                    if dados<self.prop_cruz:
                        temp=nbit1
                        nbit1=nbit2
                        nbit2=temp
                    
                    ind1+=nbit1
                    ind2+=nbit2
                    
                #print(ind1_0,ind2_0)  
                #print(ind1, ind2)
            elif tipo=='2p':
                l=self.l*self.nvars
                rs=(rnd.randrange(l),rnd.randrange(l))
                r1=min(rs)
                r2=max(rs)
                if r1==r2 or (r1==0 and r2==l-1):
                    ind1=ind1_0
                    ind2=ind2_0
                else:
                    temp=ind1_0[r1:r2]
                    ind1=ind1_0[:r1]+ind2_0[r1:r2]+ind1_0[r2:]
                    ind2=ind2_0[:r1]+temp+ind2_0[r2:]
                    #print temp,r1,r2,l
                    #print ind1_0,ind2_0,(len(ind1_0),len(ind2_0))
                    #print ind1,ind2,(len(ind1),len(ind2))
            pob1.append(ind1);pob1.append(ind2)
        #print('Cruzamiento lpob:',len(pob1))
        return(pob1)


    def mutacion(self, pob):
                #print('\Mutación')
        invert={'0':'1','1':'0'}
        
        pob_mut=list()
        for ind in pob:
            ind_mut=''
            for bit in ind:
                nbit=bit
                if rnd.random()<self.prob_mut:
                    nbit=invert.get(bit)
                ind_mut+=nbit
                
            pob_mut.append(ind_mut)
            
        return(pob_mut)
    
    def elitismo(self, fit, fit1):
        fit_max=fit[0][0]
        #print('Elitismo:',len(fit1[0]),len(fit1[1]))
        for ind in range(self.Nind):
            if self.max:
                if fit1[ind][0]>fit_max or rnd.random()>self.elit:
                    continue
                if fit1[ind][0]<fit[ind][0]:
                    fit1[ind][0]=fit[ind][0]
                    fit1[ind][1]=fit[ind][1]
                    
            elif not self.max:
                if fit1[ind][0]<fit_max or rnd.random()>self.elit:
                    continue
                if fit1[ind][0]>fit[ind][0]:
                    fit1[ind][0]=fit[ind][0]
                    fit1[ind][1]=fit[ind][1]
        
        return(fit1)
    

    def start(self):
        prueba=list()
        for p in range(self.pruebas):
            optimxgen=None
            ##Creación de individuos!!!!!!!!!!!!!!!!!!!!!
            if self.deb:print('\nCreación de individuos')
            self.pob=self.crearPob()
    
            
            ##Acomodo de acuerdo al desempeño!!!!!!!!!!!!
            if self.deb:print('Evaluando individuos... ', end = '')
            fit=self.fitnes(self.pob)
            if self.deb:print('Evaluados')
            #print('5 best Fitneesss ',fit[:5])
            
        
            if self.deb: print('\nPrueba ',p+1)
                
            for gen in range(self.Ngen):
                
                #if self.deb and gen%int(self.Ngen*.1)==1:print 'Generacion:{}'.format(gen)
                ##Se hace una prueba de rigidez, para ver si ha avanzado el algoritmo
                if gen%int(self.Ngen*.3)==1:
                    if optimxgen and optimxgen==mejor[1]:
                        if self.deb: print('Se rompe en',optimxgen)
                        break
                    optimxgen=mejor[1]
                    
                if self.deb and gen%int(self.Ngen*.1)==1:print('\nGeneracion:{} - Fitness:{:.4e}'.format(gen+1,mejor[1]))
                #print(fit)
                
                #print("\nCruzamiento")
                sel=self.seleccion(fit)
                ## Cruzamiento!!!!!!!!!!!!!!!!!!!!
                pob1=self.cruzamiento(self.tipo_cruz,sel)
        
                ##Mutación!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
                pob_mut=self.mutacion(pob1)

                ##se vuelve a probar la nueva población
                t1=time.time()
                fit1=self.fitnes(pob_mut)
                if self.deb and gen%int(self.Ngen*.1)==1:print('Tiempo para fitness por gen {:.4e}'.format(time.time()-t1))
                ### Elitismo !!!!!!!!!!!!!!!!!!!!!
                fit=self.elitismo(fit, fit1)

                #print fit[0]
                self.hist_mej[0].append(gen)
                self.hist_mej[1].append(fit[0][0])
                self.hist_mej[2].append(fit[0][1])
                mejor=(self.decodificado(fit[0][1]),fit[0][0])
                #print 'Mejor:',mejor
                #if self.deb and gen%int(self.Ngen*.1)==0:print('Mejor Individuo:',fit[0][0],' vars=',mejor[0],' f=',mejor[1] )

            
            prueba.append(mejor)
        
        prueba=sorted(prueba, key=lambda p: p[1])

        
        return(prueba[0])
            
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    def f(x,y):
        return x**2 - y**2*np.sin(y)
    
    x = np.linspace(-15,15,50)
    y = np.linspace(-15,15,50)

    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y, f(X, Y))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ag=AG()
    ag.parametros(Nind=100,Ngen=1000,optim=0)
    ag.variables(variables=[['x',-15,15],
                            ['y',-15,15]])

    
    ag.Fobj(f)
    t1=time.time()
    res=ag.start()
    print('Tiempo de cómputo {:.4e}'.format(time.time()-t1))
    print(res)

    
    plt.show()
