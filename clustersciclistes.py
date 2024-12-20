"""
@ IOC - CE IABD
"""
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

def load_dataset(path):
	"""
	Carrega el dataset de registres dels ciclistes

	arguments:
		path -- dataset

	Returns: dataframe
	"""
	df = pd.read_csv(path)

	return df

def EDA(df):
	"""
	Exploratory Data Analysis del dataframe

	arguments:
		df -- dataframe

	Returns: None
	"""
	print(df.head(5))
	print(df.describe())

	return None

def clean(df):
	"""
	Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

	arguments:
		df -- dataframe

	Returns: dataframe
	"""
	df = df.drop('dorsal', axis=1)
	df = df.drop('total', axis=1)
	print(df.head(5))
	return df

def extract_true_labels(df):
	"""
	Guardem les etiquetes dels ciclistes (BEBB, ...)

	arguments:
		df -- dataframe

	Returns: numpy ndarray (true labels)
	"""
	y = df['name']
	return y.to_numpy()

def visualitzar_pairplot(df):
	"""
	Genera una imatge combinant entre sí tots els parells d'atributs.
	Serveix per apreciar si es podran trobar clústers.

	arguments:
		df -- dataframe

	Returns: None
	"""
	sns.pairplot(df, vars = ["mu_p", "mu_b"])
	# plt.show()
	plt.savefig('img/pairplot.png')
	return None

def clustering_kmeans(data, n_clusters=4):
	"""
	Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
	Entrena el model

	arguments:
		data -- les dades: tp i tb

	Returns: model (objecte KMeans)
	"""
	kmeans = KMeans(n_clusters=n_clusters, random_state=0)
	model  = kmeans.fit(data)

	return model

def visualitzar_clusters(data, labels):
	"""
	Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

	arguments:
		data -- el dataset sobre el qual hem entrenat
		labels -- l'array d'etiquetes a què pertanyen les dades (hem assignat les dades a un dels 4 clústers)

	Returns: None
	"""
	plt.clf()
	sns.scatterplot(x="mu_p", y="mu_b", hue=labels, data=data)
	plt.savefig('img/clusters.png')
	#plt.show()

	return None

def associar_clusters_patrons(tipus, model):
	"""
	Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
	S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

	arguments:
	tipus -- un array de tipus de patrons que volem actualitzar associant els labels
	model -- model KMeans entrenat

	Returns: array de diccionaris amb l'assignació dels tipus als labels
	"""
	# proposta de solució

	dicc = {'tp':0, 'tb': 1}

	logging.info('Centres:')
	for j in range(len(tipus)):
		logging.info('{:d}:\t(tp: {:.1f}\ttb: {:.1f})'.format(j, model.cluster_centers_[j][dicc['tp']], model.cluster_centers_[j][dicc['tb']]))

	# Procés d'assignació
	ind_label_0 = -1
	ind_label_1 = -1
	ind_label_2 = -1
	ind_label_3 = -1

	suma_max = 0
	suma_min = 50000

	with open("model/clustering_model.pkl", "rb") as f:
		clustering_model = pickle.load(f)

	for j, center in enumerate(clustering_model.cluster_centers_):
		suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
		if suma_max < suma:
			suma_max = suma
			ind_label_3 = j
		if suma_min > suma:
			suma_min = suma
			ind_label_0 = j

	tipus[0].update({'label': ind_label_0})
	tipus[3].update({'label': ind_label_3})

	lst = [0, 1, 2, 3]
	lst.remove(ind_label_0)
	lst.remove(ind_label_3)

	if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
		ind_label_1 = lst[0]
		ind_label_2 = lst[1]
	else:
		ind_label_1 = lst[1]
		ind_label_2 = lst[0]

	tipus[1].update({'label': ind_label_1})
	tipus[2].update({'label': ind_label_2})

	logging.info('\nHem fet l\'associació')
	logging.info('\nTipus i labels:\n%s', tipus)
	return tipus

def generar_informes(df, tipus):
	"""
	Generació dels informes a la carpeta informes/. Tenim un dataset de ciclistes i 4 clústers, i generem
	4 fitxers de ciclistes per cadascun dels clústers

	arguments:
		df -- dataframe
		tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

	Returns: None
	"""
	df_0 = df.loc[df['label'] == 0]
	df_1 = df.loc[df['label'] == 1]
	df_2 = df.loc[df['label'] == 2]
	df_3 = df.loc[df['label'] == 3]

	df_0.to_csv('informes/BEBB.txt', sep='\t', index=False)
	df_1.to_csv('informes/MEMB.txt', sep='\t', index=False)
	df_2.to_csv('informes/BEMB.txt', sep='\t', index=False)
	df_3.to_csv('informes/MEBB.txt', sep='\t', index=False)

	logging.info('S\'han generat els informes en la carpeta informes/\n')

	return None

def nova_prediccio(dades, model):
	"""
	Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

	arguments:
		dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
		model -- clustering model
	Returns: (dades agrupades, prediccions del model)
	"""
	pred_labels_total = model.fit_predict(dades)
	#print(pred_labels_total)

	pred_labels = [pred_labels_total[-4],pred_labels_total[-3],pred_labels_total[-2],pred_labels_total[-1]]

	return dades, pred_labels

# ----------------------------------------------

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, force=True)
	path_dataset = './data/ciclistes.csv'
	"""
	TODO:
	load_dataset
	EDA
	clean
	extract_true_labels
	eliminem el tipus, ja no interessa .drop('tipus', axis=1)
	visualitzar_pairplot
	clustering_kmeans
	pickle.dump(...) guardar el model
	mostrar scores i guardar scores
	visualitzar_clusters
	"""

	df = load_dataset("data/ciclistes.csv")
	df_total = load_dataset("data/ciclistes.csv")

	EDA(df)

	df = clean(df)

	y = extract_true_labels(df)

	X = df.drop('name', axis=1)

	df = df.drop('name', axis=1)

	visualitzar_pairplot(X)

	model = clustering_kmeans(X)

	with open("model/clustering_model.pkl", "wb") as f:
		pickle.dump(model, f)


	with open("model/clustering_model.pkl", "rb") as f:
		clustering_model = pickle.load(f)

	logging.info('\nhomogeneity_score:\n%s', homogeneity_score(y, model.labels_))
	logging.info('\nv_measure_score:\n%s', v_measure_score(y, model.labels_))
	logging.info('\ncompleteness_score:\n%s', completeness_score(y, model.labels_))

	scores = {homogeneity_score(y, model.labels_)}

	dicc_scores = [
		{"homogeneity_score": homogeneity_score(y, model.labels_)},
		{"v_measure_score": v_measure_score(y, model.labels_)},
		{"completeness_score": completeness_score(y, model.labels_)}
	]

	with open('model/scores.pkl', 'wb') as handle:
		pickle.dump(dicc_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

	visualitzar_clusters(X, y)

	# array de diccionaris que assignarà els tipus als labels
	tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

	"""
	afegim la columna label al dataframe
	associar_clusters_patrons(tipus, clustering_model)
	guardem la variable tipus a model/tipus_dict.pkl
	generar_informes
	"""

	df["label"] = model.labels_
	df_total["label"] = model.labels_

	tipus_dict = associar_clusters_patrons(tipus, model)

	with open('model/tipus_dict.pkl', 'wb') as handle:
		pickle.dump(tipus_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# with open('model/tipus_dict.pkl', 'rb') as handle:
	# 	tipus_loaded = pickle.load(handle)


	generar_informes(df_total, tipus)

	# Classificació de nous valors
	# nous_ciclistes = [
	# 	[500, 3230, 1430, 4670], # BEBB 0
	# 	[501, 3300, 2120, 5420], # BEMB 2
	# 	[502, 4010, 1510, 5520], # MEBB 3
	# 	[503, 4350, 2200, 6550] # MEMB 1
	# ]

	nous_ciclistes = [
		{"mu_p": 3230, "mu_b": 1430}, # BEBB 0
		{"mu_p": 3300, "mu_b": 2120}, # BEMB 2
		{"mu_p": 4010, "mu_b": 1510}, # MEBB 3
		{"mu_p": 4350, "mu_b": 2200} # MEMB 1
	]

	X_nou = pd.concat([X, pd.DataFrame(nous_ciclistes)])
	"""
	nova_prediccio

	#Assignació dels nous valors als tipus
	for i, p in enumerate(pred):
		t = [t for t in tipus if t['label'] == p]
		logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
	"""

	pred = nova_prediccio(X_nou, model)

	for j in pred[1]:
		printed = True
		for d in tipus_dict:
			for i in d.items():
				a = d.get('name')
				b = d.get('label')
				if j == b and printed:
					print(a)
					printed = False
