import numpy as np

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