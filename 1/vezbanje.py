import numpy as np

def afinizuj(coords):
	return [coord / coords[-1] for coord in coords]

def osmoteme(p1, p2, p3, p5, p6, p7, p8):
	
	temp = afinizuj(np.cross(np.cross(p2, p6), np.cross(p1, p5)))
	Xb1 = [round(x) for x in temp]
	temp = afinizuj(np.cross(np.cross(p2, p6), np.cross(p3, p7)))
	Xb2 = [round(x) for x in temp]
	temp = afinizuj(np.cross(np.cross(p1, p5), np.cross(p3, p7)))
	Xb3 = [round(x) for x in temp]
	
	temp = afinizuj(np.cross(np.cross(p5, p6), np.cross(p7, p8)))
	Yb1 = [round(x) for x in temp]
	temp = afinizuj(np.cross(np.cross(p5, p6), np.cross(p1, p2)))
	Yb2 = [round(x) for x in temp]
	temp = afinizuj(np.cross(np.cross(p7, p8), np.cross(p1, p2)))
	Yb3 = [round(x) for x in temp]
	
	#Xb = (Xb1 + Xb2 + Xb3) / 3
	Xb = (np.rint(np.divide(np.array([Xb1, Xb2, Xb3]).sum(axis=0), 3))).astype(int).tolist()
	
	#Yb = (Yb1 + Yb2 + Yb3) / 3
	Yb = (np.rint(np.divide(np.array([Yb1, Yb2, Yb3]).sum(axis=0), 3))).astype(int).tolist()
	
	p4 = afinizuj(np.cross(np.cross(p8, Xb), np.cross(p3, Yb)))
	
	return [round(p) for p in p4]

			#p1				p2				p3				p5			p6				p7			p8
points = [[101, 162, 1], [209, 314, 1], [485, 147, 1], [85, 87, 1], [206, 231, 1], [509, 83, 1], [375, 10, 1]]
# ~ resenje priblizno [363, 72]
print(osmoteme(points[0], points[1], points[2], points[3], points[4], points[5], points[6]))
