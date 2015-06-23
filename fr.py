
import cv2
import sys
import os


import numpy as np
from numpy.linalg import inv
#imagePath = sys.argv[1]
#image = cv2.imread(imagePath)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # dont need untill get from camera 
#imagePath = sys.argv[1]


#image = cv2.imread(imagePath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
db = []
db.append([]) 
db.append([]) 
db.append([])        #NEED AMOUNT OF THESE AS PEOPLE 

db[0].append(cv2.imread('myfaces/1.jpg',0))


db[1].append(cv2.imread('yourface/1.jpg',0))
db[1].append(cv2.imread('yourface/2.jpg',0))
db[1].append(cv2.imread('yourface/3.jpg',0))
db[1].append(cv2.imread('yourface/4.jpg',0))
db[1].append(cv2.imread('yourface/5.jpg',0))
db[1].append(cv2.imread('yourface/6.jpg',0))


db[2].append(cv2.imread('uglyface/1.jpg',0))
db[2].append(cv2.imread('uglyface/2.jpg',0))
db[2].append(cv2.imread('uglyface/3.jpg',0))








n = 0
meancv = 0
for row in db:
	for column in row:
		meancv = meancv + column 
		n += 1


meancv = meancv / n          #calculate the mean 
#cv2.imshow('mean',meancv)

imMmeancv = []

for row in db:
	for column in row:
		imMmeancv.append(column - meancv)


A = np.zeros((db[0][0].shape[0], n*db[0][0].shape[1]))


for i in range(0, db[0][0].shape[1]):
	for j in range(0, db[0][0].shape[0]):
		for k in range(0,n):
			A[j][i+(db[0][0].shape[1]*k)] = imMmeancv[k][j][i]



At = np.zeros((n*db[0][0].shape[1],db[0][0].shape[0]))
At = cv2.transpose(A)

C = np.zeros((db[0][0].shape[0],db[0][0].shape[0]))
C = np.dot(A,At)

U, s, V = np.linalg.svd(C)      # svd
Ut = cv2.transpose(U)
iU = np.dot(Ut,U)
iU = inv(iU)
iU = np.dot(iU,U)              
 
q = []
q.append([])
q.append([])
q.append([])
q.append([])     #NEED AMOUNT OF THESE AS PEOPLE 
cn = 0
for i in range(0,len(db)):			#filling out db distance matrix 
	for j in range(0,len(db[i])):
		q[i].append(np.dot(iU,imMmeancv[cn]))
		cn += 1

image = cv2.imread('myfaces/3.jpg',0)     #for testing 
bT = image - meancv 		
qT = np.dot(iU,bT)	#evalues of new image



#should normalize the evectors here 

# np.amin(q[0][0], axis=0)       #max of the columns 
# np.amax(q[0][0], axis=0)       #min of the columns 
# normalization = xi - min(x) / max(x) - min(x)
for i in range(0,len(db)):			#normalizing evectos
	for j in range(0,len(db[i])):
		qmin = np.amin(q[i][j], axis=0)
		qmax = np.amax(q[i][j], axis=0)
		q[i][j] = (q
		[i][j]-qmin)/(qmax-qmin)




qTmin = np.amin(qT, axis=0)    
qTmax = np.amax(qT, axis=0)
qT = (qT-qTmin)/(qTmax-qTmin)     #normalizing input image





distq = []
distq.append([])
distq.append([])
distq.append([])
distq.append([])    #NEED AMOUNT OF THESE AS PEOPLE 
for i in range(0,len(db)):		#calculating the distance of new image
	for j in range(0,len(db[i])):
		distq[i].append(np.sqrt((qT - q[i][j])**2))


dis = []
dis.append([])
dis.append([])
dis.append([])
dis.append([])
for i in range(0,len(distq)):			#this adds up the rows of the evector 
	for j in range(0,len(distq[i])):    #then places the sum into a new matrix, this is the dis of the evector 
		c = 0							#this is done only for n evectors, which contain the most usable data
		for k in range(0,n):
			for l in range(0,distq[i][j].shape[0]):
				c += distq[i][j][l][k]
		dis[i].append(np.sqrt(c))
	


vote = []
for i in range(0,len(dis)):
	sm = 0
	for j in range(0,len(dis[i])):
		sm += dis[i][j]	
	vote.append(sm/len(dis[i]))     #take the average 


name = vote.index(min(vote))    #chooses the faces with least





cv2.waitKey(0)
