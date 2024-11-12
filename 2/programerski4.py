import numpy as np
from numpy import linalg  #zbog SVD algoritma
np.set_printoptions(precision=5, suppress=True)
 
 # ovde pišete pomocne funkcije, a ima ih puno jer koristite sve do sada
def afinizuj(coords):
	return [coords[0] / coords[2], coords[1] / coords[2]]

def find_norm_matrix(pts):
	
	afine_pts = [afinizuj(x) for x in pts]
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

def dveJednacine(orig, img):
	
    pt1 = np.array(orig)
    pt2 = np.array(img)
    
    nula = [0,0,0]
    
    jed1 = np.concatenate((nula, -pt2[2]*pt1, pt2[1]*pt1))
    jed2 = np.concatenate((pt2[2]*pt1, nula, -pt2[0]*pt1))
    
    return np.array([jed1, jed2])

def svd(orig, img):
	
	a = []
	for x,y in zip(orig, img):
		jed1, jed2 = dveJednacine(x,y)
		a.append(jed1)
		a.append(jed2)
	
	_, _, vt = np.linalg.svd(a)
	
	return vt[-1]

def DLT(pts1, pts2):
	
	A = svd(pts1, pts2)
	
	return A.reshape(3,3)

def DLTwithNormalization(origs, imgs):
	
	T = find_norm_matrix(origs)
	pts1 = []
	for coords in origs:
		temp = np.matmul(T, coords).tolist()
		pts1.append(temp)
	
	Tp = find_norm_matrix(imgs)
	pts2 = []
	for coords in imgs:
		temp = np.matmul(Tp, coords).tolist()
		pts2.append(temp)
	
	Pp = DLT(pts1, pts2)
	
	Tp_inv = np.linalg.inv(Tp)
	T_inv = np.linalg.inv(T)
	
	mat = np.matmul(Tp_inv, Pp)
	mat = np.matmul(mat, T)
 
	return mat * 1/mat[-1][-1]

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]] 
pravougaonik1 = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1], [2,1,5], [-16,-5,5]]
print(DLTwithNormalization(trapez, pravougaonik1))
