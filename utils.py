import numpy as np
import pandas as pd


def bestinhindsight(data):
    # returns value for index for action j \in J that's best
    return sum(data)[np.argmax(sum(data))]

def ew(data: np.ndarray, eps: float = 0.0):
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
    


class DataGeneration:

    def AFP(self, N, J):
        """
        N rounds
        J actions
        """
        vals = np.zeros((N,J))  # Nxj
        runvals = np.zeros(J)
        for i in range(N):
            payoff = np.random.sample()
            argmin = np.argmin(runvals)
            vals[i][argmin] = payoff
            runvals[argmin] += payoff

        # print('Total Payoffs:', runvals)
        return vals

    def BP(self, N, J):
        probs = np.random.uniform(0,0.5, J)
        vals = np.zeros((N,J))
        for i in range(N):
            # to vectorize if pssbl
            for j in range(J):
                # n=1 from a binomial, which is equivalent to Bernoulli.
                vals[i][j] = probs[j] if np.random.binomial(1, p=probs[j]) else 0

        return vals

    def DIW(self, year):
        """
        soccer data Premier League
        N match days, K teams: 38x20
        each round i:   
            payoff = points (3,1,0)     for if a team wins/loses

        years: 1011 -> 1920
        """
        if year == "all":
            file = pd.read_csv(f"cleaned_data/epltenyears_pts.csv")
            return file.to_numpy()
        else:
            file = pd.read_csv(f"cleaned_data/epl{str(year)}_pts.csv")
            return file.to_numpy()

    def AGM(self):
        # TODO: adversarial generative model
        pass


# dg = DataGeneration()
# a = dg.AFP(100, 5)
# dg.BP(100,5)

# bestinhindsight(a)