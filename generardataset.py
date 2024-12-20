import os
import logging
import numpy as np

def generar_dataset(num, ind, dicc):
	"""
	Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
	TODO: completar arguments, return. num és el número de files/ciclistes a generar. ind és l'index/identificador/dorsal.
	"""
	str_ciclistes = 'data/ciclistes.csv'

	try:
		os.makedirs(os.path.dirname(str_ciclistes))
	except FileExistsError:
		pass

	fout = open(str_ciclistes, "w")

	fout.write("dorsal,")
	fout.write("mu_p,")
	fout.write("mu_b,")
	fout.write("total,")
	fout.write("name\n")
	for num in range(0, num):
		irand = np.random.randint(4, size=1)[0]
		iSigmaPujada = np.random.randint(dicc[irand].get("sigma"), size=1)[0]
		iSigmaBaixada = np.random.randint(dicc[irand].get("sigma"), size=1)[0]
		logging.info("nou ciclista")
		logging.info(str(dicc[irand].get("name")))
		logging.info(str(irand))
		logging.info(str(iSigmaPujada))
		logging.info(str(iSigmaBaixada))

		fout.write(str(ind) + ","  + str(dicc[irand].get("mu_p")+iSigmaPujada) + "," + str(dicc[irand].get("mu_b") + iSigmaBaixada) + "," + str(dicc[irand].get("mu_b")+dicc[irand].get("mu_p")+iSigmaPujada+iSigmaBaixada)+ ","+ str(dicc[irand].get("name"))+ "\n")
		ind += 1

	return None

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, force=True)
	# BEBB: bons escaladors, bons baixadors
	# BEMB: bons escaladors, mal baixadors
	# MEBB: mal escaladors, bons baixadors
	# MEMB: mal escaladors, mal baixadors

	# Port del Cantó (18 Km de pujada, 18 Km de baixada)
	# pujar a 20 Km/h són 54 min = 3240 seg
	# pujar a 14 Km/h són 77 min = 4268 seg
	# baixar a 45 Km/h són 24 min = 1440 seg
	# baixar a 30 Km/h són 36 min = 2160 seg
	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]

	generar_dataset(1000, 1000, dicc)

	logging.info("s'ha generat data/ciclistes.csv")
