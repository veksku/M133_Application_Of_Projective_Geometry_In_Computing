import numpy as np
from numpy import linalg
np.set_printoptions(precision=5, suppress=True)
 
 # ovde pišete pomocne funkcije
def afinizuj(coords):
	return [coords[0] / coords[2], coords[1] / coords[2]]
 
def normMatrix(points):
	
	afine_pts = [afinizuj(x) for x in points]
	afine_pts = np.array(afine_pts)
	
	teziste =  np.mean(afine_pts, 0)
	
	translirane_pts = afine_pts - teziste
	
	norme = [np.linalg.norm(x) for x in translirane_pts]
	prosek_normi = np.mean(norme)
