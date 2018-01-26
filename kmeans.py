import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show(img):
	imgplot = plt.imshow(img.reshape(231,195), cmap='gray')
	plt.show()

def showCenter(ver, hor, center):
	R = []
	L = []

	for i in xrange(ver):
		for j in xrange(hor):
			img = np.ones((ver,hor))*center[i*hor+j]
			L.append(img)
		R.append(np.hstack(L))
		L = []
	A = np.vstack(R)

	imgplot = plt.imshow(A, cmap='gray')
	plt.show()

def findClusterCenters(dataset, k):

	data = np.reshape(dataset,-1)

	# center random without repetation
	center = rd.sample(xrange(256),k)
	center = np.asarray(center).reshape((k,1))

	distance = np.abs(data - center)
	arg_min = np.argmin(distance,axis=0)

	old_center = np.full((k,1),np.inf)

	while np.linalg.norm(old_center-center) > 0:

		old_center[:] = center
		for i in xrange(k):
			center[i] = np.mean(data[arg_min == i])

	return center

def kmeansCompress(image,center):
	
	image2 = np.zeros(image.shape)
	image2[:] = image

	distance = np.abs(image2 - center)
	arg_min = np.argmin(distance,axis=0)
	for i in xrange(center.shape[0]):
		image2[arg_min == i] = center[i]

	return arg_min, image2

dataset = np.load('FaceImages.npy')
image = [dataset[0,:],dataset[1,:]]
karray = [16,32,64]

for i in range(len(image)):
	for j in range(len(karray)):
		center = findClusterCenters(dataset, karray[j])

		comp, krecon = kmeansCompress(image[i],center)

		show(image[i])
		show(krecon)

		if karray[j] == 16:
			showCenter(4,4,center)
		elif karray[j] == 32:
			showCenter(4,8,center)
		elif karray[j] == 64:
			showCenter(8,8,center)

