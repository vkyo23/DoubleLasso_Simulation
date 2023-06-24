# Load libraries
import numpy as np
import polars as pl 
import re
from pyomo.environ import * 
from sklearn.linear_model import LassoCV
import csv

# Set seed
np.random.seed(1)

# Sample size
J = 10
T = 10
I = 100
N = J * T
Sim = 100

# Load data
df = pl.read_csv('data/simdata_K200.csv')

# SOLVER
SOLVER = 'ipopt'

# Hyper parameter
SIGMA = 0.5

# Converting function
def mat_to_dics (mat):
    dics = {}
    for j in range(mat.shape[0]):
        for i in range(mat.shape[1]):
            dics.update({(j + 1, i + 1): mat[j, i]})
    return dics

## array to discsionary
def arr_to_dics (arr):
    dics = {}
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    for i in range(len(arr)):
        dics.update({i + 1: arr[i, 0]})
    return dics

# ----- START SIMULATION ----- #

result = np.zeros((3, Sim))
for sim in range(Sim):

    print('Simulation =', sim + 1, end = '　')
    
    # Subsetting
    tmpdata = df.filter(pl.col('simulation') == sim + 1)

    # Lasso
    lasso = LassoCV(alphas = 10 ** np.arange(-6, 1, 0.1), cv = 5,  fit_intercept = True, max_iter = 5000)
    D = tmpdata.select(pl.col('log_share')).to_numpy().reshape(N)
    XX = tmpdata.select(pl.col('^X\d+$')).drop('X1')
    lasso.fit(XX, D)

    # Extract  non-important variables
    ximp = np.array(np.where(lasso.coef_ == 0))
    ximp = ximp.reshape(ximp.shape[1])
    xnam = XX.columns
    drop = []
    for i in range(len(ximp)):
        drop.append(xnam[ximp[i]])

    # Drop the variables
    tmpdata = tmpdata.drop(drop)

    # Get the number of Post-single lasso variables
    K = tmpdata.select(pl.col("^X\d+$")).shape[1] - 1

    # Generate nu from standard normal distribution (I x K + 1 x T array)
    nu = np.zeros((I, K + 1, T))
    for t in range(T):
        nu[:, :, t] = np.random.randn(I, K + 1)
    # Calculate MU (N x I)
    MU = np.zeros((N, I))
    for i in range(I):
        tmp = np.zeros((J, T))
        for t in range(T):
            for j in range(J):
                for k in range(K+1):
                    tmp[j, t] += SIGMA * nu[i, k, t]
        MU[:, i:i+1] = tmp.reshape((N, 1))

    # Sj / S0 (N array)
    market_id = tmpdata['t'].to_numpy()
    market_id = np.array(market_id)
    S = tmpdata['S'].to_numpy()
    S = np.array(S)
    SjS0 = np.zeros(N)
    for t in range(T):
        tindex = (market_id == t + 1)
        St = S[tindex]
        SjS0[tindex] = St / (1 - np.sum(St))
    SjS0 = SjS0.reshape((N, 1))

    # Get variables
    ## X
    X = tmpdata.select(pl.col('^X\d+$'), -pl.col('P'))
    X = np.array(X)

    ## Z
    Z = tmpdata.select(pl.col('^X\d+$'), pl.col('^Z\d+$'))
    Z = np.array(Z)

    # Pyomo setting
    ## Number of X
    XNUM = np.shape(X)[1]
    ZNUM = np.shape(Z)[1]

    ##  Vars and constriants set
    XSET = range(1, XNUM + 1)
    ZSET = range(1, ZNUM + 1)
    ISET = range(1, I + 1)
    NSET = range(1, N + 1)

    ## Weighting matrix
    W = np.zeros((ZNUM, ZNUM))
    for n in range(N):
        W += np.outer(Z[n, :], Z[n, :].T)
    W = np.linalg.inv((1 / N) * W)

    # Model setting
    ## Concrete model
    model = ConcreteModel()

    ## Data
    model.X = mat_to_dics(X)
    model.Z = mat_to_dics(Z)
    model.MU = mat_to_dics(MU)
    model.W = mat_to_dics(W)
    model.SjS0 = arr_to_dics(SjS0)

    ## Parameter
    model.theta = Var(XSET)
    model.m = Var(ZSET)
    model.XI = Var(NSET)

    # Optimization functions
    ## Objective function
    def obj_rule (model):
        return sum(model.m[zi] * sum(model.m[zj] * model.W[zi, zj] for zj in ZSET) for zi in ZSET)
    model.obj = Objective(rule = obj_rule, sense = minimize)

    ## Constraint 1
    def con_rule1 (model, zi):
        return model.m[zi] == (1 / N) * sum(model.Z[n, zi] * model.XI[n] for n in NSET)
    model.con1 = Constraint(ZSET, rule = con_rule1)

    ## Constraint 2
    def con_rule2 (model, n):
        return (1/I) * sum(exp(sum(model.theta[xj] * model.X[n, xj] for xj in XSET) + model.XI[n] + model.MU[n, i]) for i in ISET) - model.SjS0[n] == 0  
    model.con2 = Constraint(NSET, rule = con_rule2)

    # Implementing optimization
    opt = SolverFactory(SOLVER)
    results = opt.solve(model)

    print('Optimization done!', end = ': ')
    print('alpha =', model.theta.get_values()[K+2])

    # Record the result
    result[0, sim:sim+1] = model.theta.get_values()[1]
    result[1, sim:sim+1] = model.theta.get_values()[K+2]
    result[2, sim:sim+1] = model.obj()

# SAVE
np.mean(result, axis = 1)
np.std(result, axis = 1)
f = open('output/result_single_K200.csv', 'w')
writer = csv.writer(f, lineterminator = '\n')
writer.writerows(result)
f.close()
