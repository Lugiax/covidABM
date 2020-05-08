import fiona
import csv
import pandas as pd
import pickle


datos = {}

#Lectura de los límites
#with fiona.open('Limites/gadm36_MEX_1.shp') as shape:
#    for d in shape:
#        if d['properties']['NAME_1'] == 'Yucatán':
#            datos['Yucatán'] = {'limites': d['geometry']['coordinates']}

with fiona.open('Limites/gadm36_MEX_2.shp') as shape:
    print(shape[0]['properties'])
    for d in shape:
        if d['properties']['NAME_1'] == 'Yucatán':
            datos[d['properties']['NAME_2']] = {'limites': d['geometry']['coordinates']}

lugares = sorted(datos.keys())
print(lugares)
print(len(lugares))
#Lectura de las coordenadas
with open('coordenadas.csv', 'r', encoding='utf-8') as f:
    print('Se leen las coordenadas del centro')
    lines = csv.reader(f)
    contador = 0
    for line in lines:
        ##0:Entidad,1:Num_Ent,2:Municipio,3:Num_Mun,4:lat,5:lon
        if line[0]=='Yucatán':
            if line[2]=='Dzán': line[2] = 'Dzan'
            elif line[2]=='Tinum': line[2] = 'Tinúm'
            elif line[2]=='Tixmehuac': line[2] = 'Tixméhuac'
            if line[2] in datos.keys():
                datos[line[2]]['centro'] = (float(line[4]), float(line[5]))
                datos[line[2]]['num'] = int(line[3])
                contador+=1
            else:
                print(f'\tEl lugar {line[2]} no se encontró')
print(f'\tSe han asignado {contador} coordenadas')


#Lectura de la población
print('Se leen los datos de población')
df = pd.read_excel('poblacion.xlsx', header = None)
df = df.iloc[:,:3]
pob = df[df.iloc[:,1]=='Población']
contador = 0
for i in range(pob.shape[0]):
    nombre = pob.iloc[i,0][4:].strip()
    if nombre=='Dzán': nombre = 'Dzan'
    elif nombre=='Tinum': nombre = 'Tinúm'
    elif nombre=='Tixmehuac': nombre = 'Tixméhuac'
    if nombre in datos.keys():
        datos[nombre]['pob'] = int(pob.iloc[i,2])
        contador+=1
    else:
        print(f'\tEl lugar {nombre} no se encontró')
print(f'\tSe han asignado {contador} datos')

with open('datos.pk', 'wb') as f:
    pickle.dump(datos, f)
