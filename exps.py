from utils import *

dg = DataGeneration()

N = 100; K = 3;
inf = 10000000
etheo = np.sqrt(np.log(K)/N)
# print(etheo)


def run(dataset):
    bih = bestinhindsight(dataset)
    # FTL
    ew_payoff_einf = ew(dataset, inf)
    # Random Guess
    ew_payoff_e0 = ew(dataset, 0)
    # TheoBest
    ew_payoff_etheo = ew(dataset, etheo)
    print(f"BIH_Payoff: {bih}, EW_einf: {ew_payoff_einf}, EW_e0: {ew_payoff_e0}, EW_etheo: {ew_payoff_etheo}")

"""
Running EW with eps=inf => killing the algo (aka, expect should be 0)
*note, but sometimes it gets payoff simply from 1st round by the random choice
so its E[EW_e=inf] = E[h]/J     where h is 1, then E[h]=0.5. when MonteCarlo-ing

eps=0,  worstcase same as EW - first choice
        best case = BIH
        "it randomly chose the one that just got subsidy for every round,
        every other alternate choice is worse"
"""

afp = dg.AFP(N,K)
# run(afp)


bp = dg.BP(10, 3)
# run(bp)

pl = dg.DIW("all")
# print(pl)
run(pl)