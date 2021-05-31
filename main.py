import sys
import numpy as np

from numpy.linalg import matrix_rank
from scipy.spatial.distance import cdist

from cur import algorithmCUR
from UVdec import UV_dec

def rmse(prediction, y):
    return np.sqrt((np.square(prediction - y)).mean(axis=None))

def frobenius(pred, Y):
    return np.sum(pow(Y - pred,2))

def SVD(T_mat,rank):
    U, sigma, V = np.linalg.svd(T_mat, full_matrices=True)
    return np.matmul(np.matmul(U[:,:rank], np.diag(sigma[:rank])), V[:rank,:])

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
    
    uv_pred = recommend(uv_mat,test)
    svd_pred = recommend(svd_mat,test)
    cur_pred = recommend(cur_mat,test)    
