import numpy as np
from numpy.linalg import svd, pinv

def selectElem(label,A,rank,eps=1):
	u,sigma,v = svd(A,full_matrices=True)
	colprob = [(1/rank)*np.sum(pow(col,2)) for col in v[:rank,:].T]
	c = rank*np.log(rank)/pow(eps,2)
	colToKeep = [indx for indx,prob in enumerate(colprob) if min(1,c*prob) == 1]
	if label != 'col':
		return A[:,colToKeep].T
	return A[:,colToKeep]   

def algorithmCUR(A,rank,eps=1):
	C = selectElem('col',A,rank,eps)
	R = selectElem('row',A.T,rank,eps)
	U = np.dot(np.dot(pinv(C),A),pinv(R))
	pred = np.dot(np.dot(C,U),R)
	return pred