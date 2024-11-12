import numpy as np
from numpy import linalg  #zbog SVD algoritma
np.set_printoptions(precision=5, suppress=True)
 
 # ovde pišete pomocne funkcije
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
	
	uu, dd, vt = np.linalg.svd(a)
	
	return vt[8]

def DLT(pts1, pts2):
	
	A = svd(pts1, pts2)
	koef_norme = 1/A[-1]
	
	return A.reshape(3,3) * koef_norme
 
 