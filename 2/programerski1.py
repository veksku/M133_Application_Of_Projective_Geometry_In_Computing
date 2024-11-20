import numpy as np
np.set_printoptions(precision=5, suppress=True)
 
###################################################################

def formiraj_matricu(A, B, C, alfa, beta, gama):
	
	row1 = A*alfa
	row2 = B*beta
	row3 = C*gama
	
	M = np.transpose(np.array([row1, row2, row3]))
	
	return M
	
def proveri_kolinearnost(alfa, beta, gama):
	
	alfa = np.round(alfa, 7) 
	alfa = np.where(alfa ==0 , 0.0 , alfa)
	
	beta = np.round(beta, 7) 
	beta = np.where(beta ==0 , 0.0 , beta)
	
	gama = np.round(gama, 7) 
	gama = np.where(gama ==0 , 0.0 , gama)
	
	return any([alfa == 0, beta == 0, gama == 0])

def naivni(origs, imgs):
	
	A, B, C, D = np.array(origs[0]), np.array(origs[1]), np.array(origs[2]), np.array(origs[3])
	M1 = np.transpose([A, B, C])
	alfa, beta, gama = np.linalg.solve(M1, D)
	
	if proveri_kolinearnost(alfa, beta, gama):
		return "Losi originali!"
	P1 = formiraj_matricu(A, B, C, alfa, beta, gama)
	
	Ap, Bp, Cp, Dp = np.array(imgs[0]), np.array(imgs[1]), np.array(imgs[2]), np.array(imgs[3])
	M2 = np.transpose([Ap, Bp, Cp])
	alfap, betap, gamap = np.linalg.solve(M2, Dp)
	
	if proveri_kolinearnost(alfap, betap, gamap):
		return "Lose slike!"
	P2 = formiraj_matricu(Ap, Bp, Cp, alfap, betap, gamap)
	
	P1_inv = np.linalg.inv(P1)
	
	P = P2 @ P1_inv
	
	P = np.round(P, 7) 
	P = np.where(P ==0 , 0.0 , P)
	
	return P * 1/P[-1][-1]

###################################################################

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1]] 
pravougaonik = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1]]
# ~ naivni(trapez, pravougaonik)
print(naivni(trapez, pravougaonik))
# ~ resenje:
# ~ [[ 1.   0.   0. ]
# ~ [ 0.   1.  -0.5]
# ~ [ 0.  -0.5  1. ]]

origs = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1]] 
imgs = [[- 2, - 5, 1], [2, - 5, 1], [2, 1, 1], [6, -3, 3]]   #primetite da nisu u opstem polozaju
# ~ naivni(origs, imgs)
print(naivni(origs, imgs))
# ~ resenje:
# ~ Lose slike!
