from distances_function import distance_euclidean, distance_euclidean_t, distance_edr, distance_dtw, distance_twed
import numpy as np

def get_similarity_euc(c1,c2):
    t1 = np.array(c1)
    t2 = np.array(c2)
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(-1, 1)
    return distance_euclidean_t(t1,t2)[-1][-1]
    
    

def get_similarity_edr(c1,c2,epsilon):
    t1 = np.array(c1)
    t2 = np.array(c2)
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(-1, 1)
    return distance_edr(t1,t2,epsilon)


def get_similarity_dtw(c1,c2):
    t1 = np.array(c1)
    t2 = np.array(c2)
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(-1, 1)
    return distance_dtw(t1,t2)

    

def get_similarity_twed(c1,c2,nu,_lambda):
    t1 = np.array(c1)
    t1_=list(i for i in range(len(t1)))
    t2 = np.array(c2)
    t2_=list(i for i in range(len(t2)))

    return distance_twed(t1,t1_,t2,t2_,nu,_lambda)[0]
