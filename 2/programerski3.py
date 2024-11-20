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
	
	skalirane_norme = np.sqrt(2)/prosek_normi
	
	G = np.array([[skalirane_norme, 0, 0],[0, skalirane_norme, 0],[0,0,1]])
	S = np.array([[1,0,-teziste[0]],[0,1,-teziste[1]],[0,0,1]])
	
	mat = G @ S
 
	return np.array(mat)

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]] 
print(normMatrix(trapez))
