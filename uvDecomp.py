import sys
import numpy as np
from numpy.linalg import matrix_rank, svd, pinv, norm
from scipy.spatial.distance import cdist

def rmse(prediction, y):
    return np.sqrt((np.square(prediction - y)).mean(axis=None))

def frobenius(pred, Y):
    return np.sum(pow(Y - pred,2))

def UV_dec(T_mat, rank, lr=0.01, beta=0.0002, epochs=5000):
    U = np.random.rand(T_mat.shape[0],rank)
    V = np.random.rand(rank,T_mat.shape[1])

    for epoch in range(epochs):
        for usr in range(T_mat.shape[0]):
            for itm in range(T_mat.shape[1]):
                if T_mat[usr,itm] > 0:
                    error = T_mat[usr,itm] - np.dot(U[usr,:],V[:,itm])
                    U[usr,:] = U[usr,:] + 2 * lr * (error * V[:,itm] - beta * U[usr,:])
                    V[:,itm] = V[:,itm] + 2 * lr * (error * U[usr,:] - beta * V[:,itm])
    
        pred = np.dot(U,V)
        err = 0
        for usr in range(T_mat.shape[0]):
            for itm in range(T_mat.shape[1]):
                if T_mat[usr,itm] > 0:
                    err += pow(T_mat[usr,itm] - np.dot(U[usr,:],V[:,itm]),2) + \
                           (beta/2) * np.sum((pow(U[usr,:],2) + pow(V[:,itm],2)))
       
        #print('At epoch {0} Train RMSE is {1}: '.format(epoch, err))
        if err < 0.1:
            break
    return pred

def SVD(T_mat,rank):
    U, sigma, V = np.linalg.svd(T_mat, full_matrices=True)
    return np.matmul(np.matmul(U[:,:rank], np.diag(sigma[:rank])), V[:rank,:])


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

def normalization(mat):
    for row in mat:
        print(row)
        row = row - row[row != 0].mean()
        print(row)
    return mat
   
def recommend(P,test):
    index = np.random.choice(test.shape[0], 1, replace=False)
    testUsr = test[index,:]
    itemsToPredict = [indx for indx, value in enumerate(np.nditer(testUsr)) if value == 0]
    if not itemsToPredict:
        print('Nothing to predict for user {0}, try with another one!'.format(index))
        sys.exit(1)
    
    P = normalization(P)
    testUsr = normalization(testUsr)
    print(testUsr)
    sut
    simList = [(1 - cdist(P[:,indItm].reshape(1,-1), col.reshape(1,-1), 'cosine'),indCol) for indItm in itemsToPredict
                                                                                          for indCol, col in enumerate(P.T) if ((1 - cdist(P[:,indItm].reshape(1,-1), col.reshape(1,-1), 'cosine')) > 0.75)]
    neighbours = [(sim,indx) for sim, indx in simList if sim != 1]
    
    numerator, denominator = 0, 0
    for sim, indx in neighbours:
        numerator += sim*testUsr[0][indx]
        denominator += sim
    prediction = (numerator/denominator) + testUsr[testUsr != 0].mean()
    print('mean:',testUsr[testUsr != 0].mean())

    return prediction
        
if __name__ == '__main__':
    
    test = [
        [0,3,2,1],
        [4,1,3,0],
        [1,0,4,5],
        [1,2,0,4],
        [0,1,5,4],
    ]

    M = [
        [5,3,2,1],
        [4,1,3,3],
        [1,0,4,5],
        [1,2,2,4],
        [3,1,5,4],
    ]

    M = np.array(M)
    test = np.array(test)
    #rank = matrix_rank(M)
    # Load utility matrix
    #M = np.load('utility_mat.npy')
    rank = matrix_rank(M)
    if rank > min(M.shape[0],M.shape[1]):
        print('Rank must be << than min(#rows,#cols)')
        sys.exit(1)
    
    uv_mat = UV_dec(M, rank)
    svd_mat = SVD(M,rank)
    cur_mat = algorithmCUR(M,rank)
    print('UD-Decomposition:',rmse(uv_mat,M))
    print('SVD:',rmse(svd_mat,M))
    print('CUR:',rmse(cur_mat,M))
    pred = recommend(cur_mat,test)    
