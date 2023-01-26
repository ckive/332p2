import numpy as np


def bestinhindsight(data):
    # returns value for index for action j \in J that's best
    return sum(data)[np.argmax(sum(data))]

def ew_ftl(data):
    N, J = data.shape
    h = data.max()
    ftlpayoff = np.random.choice(data[0])       # randomly choose first 1
    cumpayoffs = data[0].copy()
    for i in range(1, N):
        choice = np.argmax(cumpayoffs)          # make choice
        ftlpayoff += data[i][choice]
        cumpayoffs += data[i]                   # learn for next round

    return ftlpayoff
    

def ew(data: np.ndarray, eps: float = 0.0):
    if eps > 100:
        # run follow the leader
        return ew_ftl(data)

    N, J = data.shape
    h = data.max()
    probs = np.ones((N,J))      # easy for plotting
    # data_norm_by_h = np.divide(data, h)
    # V_ij = np.cumsum(data_norm_by_h, axis=0)      # NxJ
    V_ij = np.cumsum(data, axis=0)      # NxJ
    runpayoff = np.random.choice(data[0])     # randomly choose the first one
    actions = range(J)
    
    # print("First Row", data[0])
    for i in range(1, N):
        probs[i] = np.divide(np.power(1+eps, V_ij[i-1]/h), np.sum(np.power(1+eps, V_ij[i-1]/h)))
        
        # here @82, highest weight can be chosen but not guaranteed so not argmax.
        choice = np.random.choice(actions, p=probs[i])
        runpayoff += data[i][choice]
    
    return runpayoff
    