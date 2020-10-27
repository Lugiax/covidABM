import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle as pk
from datetime import datetime


class AnalizadorMunicipios:
    def __init__(self, path = 'Datos/datos_municipios.pk'):
        self.datos = pd.read_pickle(path)
        self.municipios = self.datos.Municipio.values
        self.generar_coordenadas()
        self.datos['densidad'] = np.log2(self.datos.Poblacion / self.datos.area)*9

    def generar_coordenadas(self, norm = True):
        coord = self.datos[['lat','lon']].values
        if norm:
            #coord = coord[::-1]
            esq1 = np.array([19.960423, -90.462547])
            esq2 = np.array([21.662516, -87.562156])
            delta = esq2-esq1
            self.datos['coord'] = [(n1, n2) for n1, n2 in (coord-esq1)/delta]
        else:
            self.datos['coord'] = [(n1, n2) for n1, n2 in coord]

    def obtener_densidades(self):
        return self.datos.densidad

    def obtener_densidad(self, num_o_nom):
        if isinstance(num_o_nom, str):
            num_o_nom = self.obtener_numero(num_o_nom)
        return self.datos.densidad.loc[num_o_nom]

    def obtener_poblacion(self, num_o_nom = None):
        if num_o_nom is None:
            return self.datos.Poblacion.sum()
        elif isinstance(num_o_nom, str):
            num_o_nom = self.obtener_numero(num_o_nom)
        return self.datos.Poblacion.loc[num_o_nom]


    def obtener_area(self, num_o_nom):
        if isinstance(num_o_nom, str):
            num_o_nom = self.obtener_numero(num_o_nom)
        return self.datos.area.loc[num_o_nom]

    def obtener_coordenadas(self):
        return self.datos.coord

    def obtener_nombre(self, num):
        return self.datos.Municipio.loc[num]

    def obtener_nombres(self, num):
        return self.datos.Municipio.loc[num]

    def obtener_numero(self, nombre):
        return self.datos[self.datos.Municipio==nombre].index[0]

    def obtener_numeros(self, nombres):
        return self.datos.where(self.datos.Municipio in nombres)

class GraficadorSimple:
    def __init__(self, data):
        if isinstance(data, str):
            self.data = pd.read_pickle(data).groupby('Fecha').mean()
        else:
            self.data = data.groupby('Fecha').mean()

def graficar(self, figsize = (10,5), ind_x_agente = 10):
    historico = leer_historico(solo_activos = True,
                            ind_x_agente = ind_x_agente)
    historico = historico.sum(axis=0)
    fechas_hist = historico.index.get_level_values(0).values

    fig, ax = plt.subplots(figsize = figsize)
    ax.plot_date(self.data.index.values, self.data.Infectados,
                'k-', label = 'Infectados sim')
    ax.plot_date(fechas_hist, historico.values,
                'r.', label = 'Infectados reales')

    plt.show()



class GeneradorMovilidad:
    def __init__(self, semanal=True,
    prediccion = True,
    semanas_a_agregar = 4,
    valor_de_relleno = None, ##En porcentaje
    path='Datos/Global_Mobility_Report.csv',
    ):
        #from sklearn.neural_network import MLPRegressor 
        self.semanal = semanal ##Por el momento solo será semanal
        self.semanas_a_agregar = semanas_a_agregar
        self.valor_de_relleno = valor_de_relleno
        self.lectura = self.leer_movilidad(path)
        self.semana_max = self.lectura.index.values.max()
        #self.predictor = MLPRegressor((50,), alpha = 0.001) #Para agregar después

    def leer_movilidad(self, path):
        #https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=57b4ac4fc40528e2
        df = pd.read_csv(path, sep =',')
        df = df[df.country_region == 'Mexico']
        df = df[df.sub_region_1 == 'Yucatan']
        df['tot'] = df.iloc[:,4:].mean(axis=1)
        df['weeknumber'] = pd.to_datetime(df.date).apply(lambda x: x.isocalendar()[1])
        return df.groupby('weeknumber').mean()['tot']

    def generar(self, semana):
        if semana <= self.semana_max:
            return self.lectura.loc[semana]
        else:
            #self.predictor.fit(self.lectura.index.values, self.lectura.values)
            nuevas_semanas = [self.semana_max+i+1 for i in range(self.semanas_a_agregar)]
            self.semana_max = nuevas_semanas[-1]
            relleno = self.valor_de_relleno if self.valor_de_relleno is not None else self.lectura.values[-1]
            valores = [relleno for _ in range(self.semanas_a_agregar)]
            self.lectura = self.lectura.append(pd.Series(valores, index = nuevas_semanas))
            return relleno


def convertir_municipio():
    datos = pd.read_pickle('Datos/datos_municipios.pk')
    return datos.Municipio

def leer_historico(solo_activos = False, intervalo = None, ind_x_agente = 1):
    path = 'Datos/historial.xlsx'
    df = pd.read_excel(path, header=[0,1]).fillna(np.uint(0))
    df.set_index(('NUM','num'), inplace=True)
    #df.drop(columns=('ENTIDAD','entidad'), inplace=True)
    #df = df.astype('uint')
    #print(df.iloc[: -21])
    #print(df.head(10))
    df.iloc[:,1:] = df.iloc[:,1:]/ind_x_agente
    if solo_activos:
        df = df.iloc[:, df.columns.get_level_values(1)=='Activos']
    if intervalo is not None:
        dia_cero, dia_final = intervalo
        df = df.iloc[:, df.columns.get_level_values(0)>=dia_cero]
        df = df.iloc[:, df.columns.get_level_values(0)<=dia_final]
    return df

def convertir_corrida(corrida, con_fechas=False):
    fechas=None
    if isinstance(corrida, str):
        datos = extraer_pk(corrida)
        corrida = datos['corrida']
        fechas_restricciones = datos['extra']['dias_de_restricciones']
    corrida.reset_index(drop=True, inplace=True)
    totales = corrida.iloc[:,1:5].values
    fechas = corrida.iloc[:,0].values
    n_fil, n_col = corrida.shape
    conteos = np.zeros((n_fil, (n_col-5)*4), dtype = np.uint)
    #import pdb; pdb.set_trace()
    for i in range(n_fil):
        for j, reg in enumerate(corrida.columns.values[5:]):
            try:
                conteos[i,j*4:(j+1)*4] = np.array(corrida.loc[i, reg])
            except:
                print(f'Error en {i}, {j}, {reg}')
    conteos_cols = pd.MultiIndex.from_product([corrida.iloc[:,5:].columns.values,
                                              ['S','E','I','R']])
    totales_cols = pd.MultiIndex.from_product([['Total'],
                                              ['S','E','I','R']])
    df_tot = pd.DataFrame(totales, columns = totales_cols)
    df_cont= pd.DataFrame(conteos, columns = conteos_cols)
    df_fechas = pd.DataFrame(fechas, columns = [('Fecha', 'fecha')])
    datos = pd.concat((df_tot, df_cont, df_fechas), axis = 1)
    #import pdb; pdb.set_trace()
    datos = datos.groupby([('Fecha', 'fecha')]).mean()

    if con_fechas:
        return datos, fechas_restricciones
    else:
        return datos


def calcular_error(simu, intervalo, inds_x_agente = 5):
    """
    Requiere los datos de la simulación, sin tratar
    el intervalo de tiempo (dia_cero, dia_final) (datetime)
    el número de individuos por agente.

    Devuelve una dataframe del error por día por municipio
    """
    return 0
    if isinstance(simu, str):
        simu = pd.read_pickle(simu)
    dia_cero, dia_final = intervalo
    un_dia = datetime.timedelta(days = 1)
    n_dias = int((dia_final-dia_cero)/un_dia)+1

    simu = simu[simu['Fecha']>=dia_cero]
    simu = simu[simu['Fecha']<=dia_final]
    fechas = simu.Fecha
    simu = convertir_corrida(simu)
    assert simu.shape[0] >= n_dias, f'{simu.shape[0]} < {n_dias}'

    simu = simu.iloc[:n_dias, simu.columns.get_level_values(1)=='I']
    simu = pd.DataFrame(simu.iloc[:,1:].values.T, 
        index = simu.iloc[:,1:].columns.get_level_values(0),
        columns = simu.index)
        #list(map(lambda x: dia_cero+un_dia*x, list(simu.index))))

    hist = leer_historico(solo_activos = True,
                            ntervalo = (dia_cero, dia_final))
    simu = simu.loc[simu.index.isin(list(hist[('ENTIDAD','entidad')]))]

    muns = list(simu.index)
    res = np.zeros(simu.shape) 
    for i, mun in enumerate(muns):
        res[i] = (hist.loc[mun].values/inds_x_agente-simu.loc[mun].values)**2
    return pd.DataFrame(res, index=simu.index, columns=simu.columns)




def promediar(archivos, plot=True, rows=2, cols=5, figsize=(20,10), title='', convertir=True):
    
    if convertir:
        print(f'Promediando los archivos disponibles ({len(archivos)}): {archivos}')
        promedio, fechas_rest = convertir_corrida(archivos[0], con_fechas=True)
    else:
        promedio = archivos[0]
        fechas_rest = None
    promedio = promedio['Total']
    fechas = promedio.index
    fechas_rest_prom = [fechas_rest[0]]
    if plot: 
        figs, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.ravel()
        axes[0].plot_date(fechas, promedio.I, 'k-')
        if fechas_rest is not None:
            axes[0].plot_date([fechas_rest[0], fechas_rest[0]],
                                [0, promedio.I.loc[fechas_rest[0]]],
                                'r--')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=90)

    for i, a in enumerate(archivos[1:]):
        nuevo, fechas_rest = convertir_corrida(a, con_fechas=True) if convertir else a['Total']
        nuevo = nuevo['Total']
        promedio = promedio.add(nuevo)
        if plot:
            axes[i+1].plot_date(fechas, nuevo.I, 'k-')
            if fechas_rest is not None:
                axes[i+1].plot_date([fechas_rest[0], fechas_rest[0]],
                                    [0, nuevo.I.loc[fechas_rest[0]]],
                                    'r--')
            fechas_rest_prom.append(fechas_rest[0])
            plt.setp(axes[i+1].xaxis.get_majorticklabels(), rotation=90)

    promedio = promedio/len(archivos)
    fechas_rest_prom = promediar_fechas(fechas_rest_prom)

    if plot:
        plt.show()
        fig_prom, ax_prom = plt.subplots(figsize=(8,5))
        ax_prom.plot_date(fechas, promedio.I, 'k-')
        ax_prom.plot_date([fechas_rest_prom, fechas_rest_prom],
                          [0, promedio.I.loc[fechas_rest_prom]],
                          'k--')
        ax_prom.set_title(title)
        ax_prom.set_xlabel('Fecha')
        ax_prom.set_ylabel('Agentes infectados')
        plt.show()

    return promedio

def extraer_pk(path):
    datos = dict()
    with open(path, 'rb') as f:
        datos['corrida'] = pk.load(f)
        datos['extra'] = pk.load(f)
    return datos

def promediar_fechas(fechas):
    #fechas es una lista con elementos datetime
    segundos = sum([f.timestamp() for f in fechas])/len(fechas)
    completa = datetime.fromtimestamp(segundos)
    fecha = datetime(completa.year, completa.month, completa.day)
    return fecha


if __name__=='__main__':
    #import datetime
    #dia_cero = datetime.datetime(2020,4,17)
    #dia_final = datetime.datetime(2020,4,27)
    #mov = GeneradorMovilidad(valor_de_relleno = -45)
    #print(mov.lectura)
    #print(mov.generar(40))
    #print(mov.generar(50))
    #print(mov.generar(60))
    #print(leer_historico())
    #print(df[(dia_cero, 'Activos')]['Valladolid'])
    #fecha = pd.Timestamp('2020-03-17')
    #print(df.iloc[:, df.columns.get_level_values(1)=='Activos'].values)
    #corrida = pd.read_pickle('resultados/simAGprueba1.pk')
    #print(calcular_error(corrida, (dia_cero, dia_final)).sum(axis=1))
    #print(convertir_corrida('resultados/simmanual1.pk'))
    #print(obtener_num_municipios().loc[25])
    #AM = AnalizadorMunicipios()
    #print(AM.obtener_nombres([1,2,3]))
    #print('Valladolid: ', AM.obtener_numero('Valladolid'))
    #print('Pob', AM.obtener_poblacion(102))
    #print('Area', AM.obtener_area(102))
    #print('Pob', AM.obtener_poblacion('Valladolid'))
    #print('Area', AM.obtener_area('Valladolid'))
    #print('Dens', AM.obtener_densidad('Valladolid'))
    #print('Poblacion total', AM.obtener_poblacion())

    #dens = AM.obtener_densidades().values
    #print(dens.max())
    #print(dens.min())
    #import matplotlib.pyplot as plt

    #plt.hist(dens, bins = 6)
    #plt.show()

    #G = GraficadorSimple('resultados/simple_5ixa_pruebasajuste12_1.pk', )
    #G.graficar(ind_x_agente = 5)
    import os
    from glob import glob
    path_carpeta = 'resultadosCasos/CuarentenaEstricta'
    corridas = [0.03]#0.007,  0.015, 0.023, 
    colors = ['darkred', 'dodgerblue', 'orange', 'mediumblue', 'indigo',
              'chocolate', 'forestgreen', 'royalblue']

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title('Distanciamiento social, sin umbral')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Agentes infectados')
    for corr, color in zip(corridas, colors):
        archivos = glob(os.path.join(path_carpeta, f'cuarentena_simple{corr}_*'))
        fechas_rest = [extraer_pk(f)['extra']['dias_de_restricciones'][0] for f in archivos]
        fecha_rest = promediar_fechas(fechas_rest)
        promedio = promediar(archivos, plot=True)
        ax.plot_date(promedio.index, promedio.I, '-', label=f'Umbral={corr}',
                    color=color)
        ax.plot_date([fecha_rest, fecha_rest],
                    [0, promedio.I.loc[fecha_rest]], '--', color=color)
    plt.legend()
    plt.show()