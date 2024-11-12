import cv2
import numpy as np
import matplotlib.pyplot as plt

# ~ def afinizuj(coords):
	# ~ return [coord / coords[-1] for coord in coords]
def afinizuj(coords):
	return [coords[0] / coords[2], coords[1] / coords[2]]

def vek_proiz(c1, c2):
	return [c1[1]*c2[2]-c1[2]*c2[1], c1[2]*c2[0]-c1[0]*c2[2], c1[0]*c2[1]-c1[1]*c2[0]]

def find_norm_matrix(pts):
	
	afine_pts = [afinizuj(x) for x in pts]
	afine_pts = np.array(afine_pts)
	
	teziste =  np.mean(afine_pts, 0)
	
	translirane_pts = afine_pts - teziste
	
	norme = [np.linalg.norm(x) for x in translirane_pts]
	prosek_normi = np.mean(norme)
	
	skalirane_norme = np.sqrt(2)/prosek_normi
	
	mat = [[skalirane_norme, 0, 0],[0, skalirane_norme, 0],[0,0,1]],[[1,0,-teziste[0]],[0,1,-teziste[1]],[0,0,1]]
	
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
	
	uu, dd, vt = np.linalg.svd(a)
	
	return vt[8]
    
def DLT(pts1, pts2):
	
	A = svd(pts1, pts2)
	
	return A

def norm_DLT(original, target):
	
	# ~ T = find_norm_matrix(original)
	# ~ print("T:")
	# ~ print(T)
	
	# ~ Tp = find_norm_matrix(target)
	# ~ print("Tp:")
	# ~ print(Tp)
	
	#DLT za odredjivanje matrice P' od ove dve normalizacije
	Pp = (DLT(original, target)).reshape(3,3)
	print("Pp:")
	print(Pp)
	
	#izlaz treba da bude P = T'^(-1) * P' * T
	
	# ~ P = np.linalg.inv(Tp) @ Pp @ T
	
	return Pp

def click_event(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		global points
		
		if len(points)<4:
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, ' ' + str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)
			cv2.circle(image, (x,y), radius=0, color=(255, 0, 0), thickness=6)
			cv2.imshow('PPGR seminarski', image)
			points.append([x, y, 1])
			# ~ print(points[-1])
		
		elif len(points)==4:
			
			#izbegavati cv2.getPerspectiveTransform(src, dst)
			#dozvoljen cv2.warpPerspective()
			# ~ print(points)
			original = np.array(points)
			target = np.array([[0, 0, 1], [300, 0, 1], [0, 300, 1], [300, 300, 1]])
			
			
			# ~ M = cv2.getPerspectiveTransform(original, target)
			M = norm_DLT(original, target)
			
			dst = cv2.warpPerspective(image, M, (300,300))
			
			cv2.imshow('PPGR seminarski', dst)
			
			#da se program ne izvrsava opet nepotrebno u narednim klikovima
			points.append([-1, -1])


path = 'slika.jpg'
image = cv2.imread(path)
window_name = 'PPGR seminarski'

cv2.imshow(window_name, image)

points = []
cv2.setMouseCallback(window_name, click_event)

#za izlaz iz prozora pritisnuti bilo koje slovo na tastaturi
cv2.waitKey(0)
cv2.destroyAllWindows()
