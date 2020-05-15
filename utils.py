def obtener_movilidad(path):
	import pandas as pd
	df = pd.read_csv(path, sep =',')
	df = df[df.country_region == 'Mexico']
	df = df[df.sub_region_1 == 'Yucatan']
	df['change'] = df.iloc[:,4:].mean(axis=1)
	df['dias'] = (pd.to_datetime(df.date) - pd.Timestamp('2020-02-15'))/pd.Timedelta('1 day')
	return df[['dias','change']].values




if __name__=='__main__':
	print(aproximador_movilidad('Datos/movilidad.csv'))
