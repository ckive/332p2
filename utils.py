import numpy as np
import pandas as pd


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
            vals[i] = np.random.binomial(1, p=probs)

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