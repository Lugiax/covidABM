import pandas as pd
import numpy as np
import datetime

def obtener_movilidad(*args):
	#https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=57b4ac4fc40528e2
	df = pd.read_csv('Datos/Global_Mobility_Report.csv', sep =',')
	df = df[df.country_region == 'Mexico']
	df = df[df.sub_region_1 == 'Yucatan']
	df['change'] = df.iloc[:,4:].mean(axis=1)
	#df['dias'] = (pd.to_datetime(df.date) - pd.Timestamp('2020-02-15'))/pd.Timedelta('1 day')
	#return df[['dias','change']].values
	df = df.set_index('date')
	return df['change']


def leer_historico(solo_activos = False, intervalo = None, ind_x_agente = 1):
	path = 'Datos/historial.xlsx'
	df = pd.read_excel(path, header=[0,1]).fillna(np.uint(0))
	df.set_index(('ENTIDAD','entidad'), inplace=True)
	if solo_activos:
		df = df.iloc[:, df.columns.get_level_values(1)=='Activos']
	if intervalo is not None:
		dia_cero, dia_final = intervalo
		df = df.iloc[:, df.columns.get_level_values(0)>=dia_cero]
		df = df.iloc[:, df.columns.get_level_values(0)<=dia_final]
	return df/ind_x_agente

def convertir_corrida(corrida):
	if isinstance(corrida, str):
		corrida = pd.read_pickle(corrida)

	totales = corrida.iloc[:,1:5].values
	fechas = corrida.iloc[:,0].values
	n_fil, n_col = corrida.shape
	conteos = np.zeros((n_fil, (n_col-5)*4), dtype = np.uint)
	for i in range(n_fil):
		for j, reg in enumerate(corrida.columns.values[5:]):
			#if i==0:
			#	print(i,j, reg)
			conteos[i,j*4:(j+1)*4] = np.array(corrida.loc[i, reg])
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
	return datos

def calcular_error(simu, intervalo, inds_x_agente = 5):
	"""
	Requiere los datos de la simulación, sin tratar
	el intervalo de tiempo (dia_cero, dia_final) (datetime)
	el número de individuos por agemte
	"""
	dia_cero, dia_final = intervalo
	un_dia = datetime.timedelta(days = 1)
	n_dias = int((dia_final-dia_cero)/un_dia)+1

	hist = leer_historico(solo_activos = True,
						intervalo = (dia_cero, dia_final))

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
	simu = simu.loc[simu.index.isin(list(hist.index))]

	muns = list(simu.index)
	res = np.zeros(simu.shape)
	for i, mun in enumerate(muns):
	    res[i] = (hist.loc[mun].values/inds_x_agente-simu.loc[mun].values)**2
	return res.sum(axis=0)

if __name__=='__main__':
	#import datetime
	#dia_cero = datetime.datetime(2020,5,9)
	#dia_final = datetime.datetime(2020,5,18)
	#print(obtener_movilidad().loc['2020-02-16'])
	#df = leer_historico()
	#print(df[(dia_cero, 'Activos')]['Valladolid'])
	#fecha = pd.Timestamp('2020-03-17')
	#print(df.iloc[:, df.columns.get_level_values(1)=='Activos'].values)
	#corrida = pd.read_pickle('resultados/sim3.pk')
	#print(calcular_error(corrida, (dia_cero, dia_final)).sum())
	print(convertir_corrida('resultados/simmanual1.pk'))
