import pandas as pd
import numpy as np


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


def leer_historico():
	path = 'Datos/historial.xlsx'
	df = pd.read_excel(path, header=[0,1]).fillna(np.uint(0))
	df.set_index(('ENTIDAD','entidad'), inplace=True)
	return df

def convertir_corrida(corrida):
	if isinstance(corrida, str):
		corrida = pd.read_pickle(path)
	
	totales = corrida.iloc[:,:5].values
	n_fil, n_col = corrida.shape
	conteos = np.zeros((n_fil, (n_col-5)*4), dtype = np.uint)
	for i in range(n_fil):
		for j in range(5,n_col-5):
			conteos[i, j*4:(j+1)*4] = np.array(corrida.iloc[i, j])
	conteos_cols = pd.MultiIndex.from_product([list(corrida.iloc[:,5:].columns),
                                              ['S','E','I','R']])
	totales_cols = pd.MultiIndex.from_product([['Total'],
                                              ['Dia','S','E','I','R']])
	df_tot = pd.DataFrame(totales, columns = totales_cols)
	df_cont= pd.DataFrame(conteos, columns = conteos_cols)
	datos = pd.concat((df_tot, df_cont), axis = 1)
	datos = datos.groupby([('Total', 'Dia')]).mean()
	return datos



if __name__=='__main__':
	print(obtener_movilidad().loc['2020-02-16'])
	df = leer_historico()
	print(df.head())
	fecha = pd.Timestamp('2020-03-17')
	print(df.iloc[:, df.columns.get_level_values(1)=='Activos'].values)
