import cv2
import numpy as np

def afinizuj(coords):
	return [coord / coords[-1] for coord in coords]

def vek_proiz(c1, c2):
	return [c1[1]*c2[2]-c1[2]*c2[1], c1[2]*c2[0]-c1[0]*c2[2], c1[0]*c2[1]-c1[1]*c2[0]]

	
def osmoteme(p1, p2, p3, p5, p6, p7, p8):
	
	temp = afinizuj(vek_proiz(vek_proiz(p2, p6), vek_proiz(p1, p5)))
	Xb1 = [round(x) for x in temp]
	temp = afinizuj(vek_proiz(vek_proiz(p2, p6), vek_proiz(p3, p7)))
	Xb2 = [round(x) for x in temp]
	temp = afinizuj(vek_proiz(vek_proiz(p1, p5), vek_proiz(p3, p7)))
	Xb3 = [round(x) for x in temp]
	
	temp = afinizuj(vek_proiz(vek_proiz(p5, p6), vek_proiz(p7, p8)))
	Yb1 = [round(x) for x in temp]
	temp = afinizuj(vek_proiz(vek_proiz(p5, p6), vek_proiz(p1, p2)))
	Yb2 = [round(x) for x in temp]
	temp = afinizuj(vek_proiz(vek_proiz(p7, p8), vek_proiz(p1, p2)))
	Yb3 = [round(x) for x in temp]
	
	#Xb = (Xb1 + Xb2 + Xb3) / 3
	Xb = (np.rint(np.divide(np.array([Xb1, Xb2, Xb3]).sum(axis=0), 3))).astype(int).tolist()
	#Yb = (Yb1 + Yb2 + Yb3) / 3
	Yb = (np.rint(np.divide(np.array([Yb1, Yb2, Yb3]).sum(axis=0), 3))).astype(int).tolist()
	
	p4 = vek_proiz(vek_proiz(p8, Xb), vek_proiz(p3, Yb))
	
	return [round(p / p4[-1]) for p in p4]
	
def click_event(event, x, y, flags, params):
	if event == cv2.EVENT_LBUTTONDOWN:
		global points
		
		if len(points)<7:
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, ' ' + str(x) + ',' + str(y), (x,y), font, 1, (255, 0, 0), 2)
			cv2.circle(image, (x,y), radius=0, color=(255, 0, 0), thickness=6)
			cv2.imshow('PPGR prvi domaci', image)
			points.append([x, y, 1])
			print([x,y])
		
		elif len(points)==7:
			x, y, _ = osmoteme(points[0], points[1], points[2], points[3], points[4], points[5], points[6])
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(image, ' ' + str(x) + ',' + str(y), (x,y), font, 1, (0, 255, 0), 2)
			cv2.circle(image, (x,y), radius=0, color=(0, 255, 0), thickness=6)
			cv2.imshow('PPGR prvi domaci', image)
			points.append([x, y, 1])
			print([x,y])


path = 'slika.jpg'
image = cv2.imread(path)
window_name = 'PPGR prvi domaci'

cv2.imshow(window_name, image)

points = []
cv2.setMouseCallback(window_name, click_event)

#za izlaz iz prozora pritisnuti bilo koje slovo na tastaturi
cv2.waitKey(0)
cv2.destroyAllWindows()
