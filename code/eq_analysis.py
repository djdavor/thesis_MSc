# Imports
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import time


### 0 = RN option
### 1 = R option
simulations, utilities, costs_r = {}, {}, {}

## INIT PARAMS ##
#### Brownian motion
points = 10000
paths = 100
mu_c = 0.0
sigma_c = 0.3 #ex: 5.0
years = 20
S_0 = 30.0
r = 0.045

#### Principal
n_agents = 2 #10
beta = 1
delta = 1
lamb = 0.5
alphas = [0.75, 1.0] #also more
gammas = [0, 0.1] #also more
thetas = [0, 1]

#### Agent (67/33)
c = 2500000
n_s = 83333
n_o = 300
rho_L = 1.5
rho_H = 2.5
U_hat = 0
a_h = 1
a_l = 0
sigma_h = 0.001
sigma_l = 0
y_RN = 10
y_R1 = 10
y_R2 = 10




## SIMULATE BROWNIAN MOTION PATH WITH CONSTANT MEAN AND STANDARD DEVIATION
def brownian_motion(S_0, a=0, sigma=0, is_RN = 0, is_R = 0): # is_RN = number of agents choosing RN option, is_R = number of agents choosing R option

    # Seed the random number generators
    rng = np.random.default_rng(42)
    rng_bis = np.random.default_rng(96) #Volatility's Brownian motion
    
    # Create the initial set of random normal draws
    Z = rng.normal(0.0, 1.0, (paths, points))
    #Z_bis = rng_bis.normal(0.0, 1.0, (paths, points))

    # Define the time step size and t-axis
    interval = [0.0, 1.0]
    dt = (interval[1] - interval[0]) / (points - 1)
    #t_axis = np.linspace(interval[0], interval[1], points)
    p_year = points / years

    # Use Equation 3.3 from [Glasserman, 2003] to sample brownian motion paths
    X = np.zeros((paths, points))
    X[:, 0] = S_0 # Set the initial value of the stock price
    for idx in range(points - 1):
        real_idx = idx + 1

        if (real_idx <= y_RN*p_year) and (real_idx <= (y_R1+y_R2)*p_year): #Both agents are exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + a * dt * (is_RN + is_R) + delta * sigma * np.sqrt(dt) + sigma_c * np.sqrt(dt) * Z[:, idx]
        
        elif (real_idx <= y_RN*p_year) and (real_idx > (y_R1+y_R2)*p_year): #Only RN agent is exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + a * dt * is_RN + delta * sigma * np.sqrt(dt) + sigma_c * np.sqrt(dt) * Z[:, idx]
            
        elif (real_idx > y_RN*p_year) and (real_idx <= (y_R1+y_R2)*p_year): #Only R agent is exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt +  a * dt * is_R + delta * sigma * np.sqrt(dt) + sigma_c * np.sqrt(dt) * Z[:, idx]

        else: #No agent is exerting additional effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + sigma_c * np.sqrt(dt) * Z[:, idx]

    # Obtain the set of final path values
    final_values = pd.DataFrame({'final_values': X[:, -1]})

    '''
    # Plot these paths
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for path in range(paths):
        ax.plot(t_axis, X[path, :])
    ax.set_title("Constant mean and standard deviation Brownian Motion sample paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("Asset Value")
    #plt.show()

    # Estimate and plot the distribution of these final values with Seaborn
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.kdeplot(data=final_values, x='final_values', fill=True, ax=ax)
    ax.set_title("Kernel Density Estimate of asset path final value distribution")
    ax.set_ylim(0.0, 0.325)
    ax.set_xlabel('Final Values of Asset Paths')
    #plt.show()
    '''

    # Output the mean and stdev of these final values
    #print(final_values.mean(), final_values.std())

    #Return the matrix of simulated paths
    return X 


## RUN BEFOREHAND SIMULATION PATHS FOR ALL POSSIBLE COMBINATIONS OF rn, r, a, sigma ##
def run_sims():
    for rn in [0, 1]:
        for r in [0, 1]:
            for a in [a_l, a_h]:
                for sigma in [sigma_l, sigma_h]:
                    simulations[(a, sigma, rn, r)] = brownian_motion(S_0, a, sigma, rn, r)
                    print("Sim done for: ", a, sigma, rn, r)
   

## COMPUTE AGENT'S UTILITY FUNCTION AS EXPECTATION OF INTEGRAL ## 
def agent_util(a, sigma, rho, theta, alpha, gamma):
    
    if theta == 0:
        sims = simulations[(a, sigma, 1, 0)]
    else:
        sims = simulations[(a, sigma, 0, 1)]

    #Compute expectation of integral
    X = np.zeros(paths)

    #Compute integral for each path
    for i in range(paths):

        I = np.zeros(points)
        p_year = points / years

        for j in range(points):

            #Compute wealth\'
            if (theta == 0) and ((j+1) > y_RN*p_year): #RN option has been exercised/expired ==> account for such cash flow
                W = n_s * sims[i, j] + c * (1 + r/(p_year))**j + n_o * max(0, sims[i, math.floor(y_RN*p_year)] - S_0) * (1 + r/(p_year))**(j-math.floor(y_RN*p_year)) #last term is cash now
            elif (theta == 1) and ((j+1) >= y_R1*p_year):
                if (j+1) <= (y_R1+y_R2)*p_year: #R option has been exercised only the first time 
                    W = n_s * sims[i, j] + c * (1 + r/(p_year))**j + (alpha*n_o) * max(0, sims[i, math.floor(y_R1*p_year)] - S_0) * (1 + r/(p_year))**(j-math.floor(y_R1*p_year)) + (1-alpha+gamma)*n_o * max(0, sims[i, j] - S_0) ##last term is still option
                else: #R option has been exercised twice
                    W = n_s * sims[i, j] + c * (1 + r/(p_year))**j + (alpha*n_o) * max(0, sims[i, math.floor(y_R1*p_year)] - S_0) * (1 + r/(p_year))**(j-math.floor(y_R1*p_year)) + (1-alpha+gamma)*n_o * max(0, sims[i, math.floor((y_R1+y_R2)*p_year)] - S_0) * (1 + r/(points/years))**(j-math.floor((y_R1+y_R2)*p_year)) #last term is also cash now               
            else: #Option has not been exercised yet
                W = n_s * sims[i, j] + n_o * max(0, sims[i, j] - S_0) + c * (1 + r/(points/years))**j

            #Compute utility of wealth and effort 
            u = (W**(1-rho)-1)/(1-rho) - 1/2*(a**2)

            #Discount by e^{-rt}
            u = u * np.exp(-r*j)

            I[j] = u
        

        X[i] = I.sum()    ####FINAL DISCOUNTING MISSING 

    exp_util = X.mean()
    return exp_util


## RUN BEFOREHAND UTILITIES FOR ALL POSSIBLE COMBINATIONS OF rn, r, a, sigma ##
def run_utils():
    for alpha in alphas:
        for gamma in gammas:
            for rho in [rho_L, rho_H]:
                for theta in thetas:
                    for a in [a_l, a_h]:
                        for sigma in [sigma_l, sigma_h]:
                            utilities[(a, sigma, rho, theta, alpha, gamma)] = agent_util(a, sigma, rho, theta, alpha, gamma)
                            print("Util done for: ", a, sigma, rho, theta, alpha, gamma)


## COMPUTE EXPECTED TERMINAL STOCK PRICE ## -- NOTE THAT SIGMA PLAYS NO ROLE HERE (REFLECTS ALSO PRINCIPAL'S RN)
def exp_terminal_stock (S_0, a_tot):

    exp_value =  S_0 * np.exp(a_tot * years) #Formula for expected value of Geomtric Brownian Motion
    return exp_value


## COMPUTE AGENT'S CHOICE OF OPTIMAL CONTROLS a, sigma ##
def agent_choice (alpha, gamma, rho, theta=1):

    util_a_h_sigma_h = utilities[(a_h, sigma_h, rho, theta, alpha, gamma)]
    util_a_h_sigma_l = utilities[(a_h, sigma_l, rho, theta, alpha, gamma)]
    util_a_l_sigma_h = utilities[(a_l, sigma_h, rho, theta, alpha, gamma)]
    util_a_l_sigma_l = utilities[(a_l, sigma_l, rho, theta, alpha, gamma)]

    util_max = max(util_a_h_sigma_h, util_a_h_sigma_l, util_a_l_sigma_h, util_a_l_sigma_l)

    #The ordering represents how ties are broken - agent prefers a_h over a_l and sigma_h over sigma_l when indifferent
    if util_max == util_a_h_sigma_h:
        return [a_h, sigma_h, util_a_h_sigma_h]
    elif util_max == util_a_h_sigma_l:
        return [a_h, sigma_l, util_a_h_sigma_l]
    elif util_max == util_a_l_sigma_h:
        return [a_l, sigma_h, util_a_l_sigma_h]
    else:
        return [a_l, sigma_l, util_a_l_sigma_l]
        


###CODE TAKEN FROM valuation_ESOs.ipynb - rn_eso & r_eso_mod
## COST OF RN OPTION ##
def rn_eso(S0=S_0, K=S_0, T=10, v=2, r=r, N=500, sigma=sigma_c, m=2):
    #Init values
    dt = 1/N                        #number of steps
    u = np.exp(sigma * np.sqrt(dt)) #using CRR method with (constant) volatility
    d = 1/u                         #to maintain the triangular structure of the tree (i.e., recombinant tree)
    q = (np.exp(r*dt) - d)/(u-d)    #q is the RN probability
    disc = np.exp(-r*dt)            #discount

  #Build up terminal stock price nodes
    S = np.zeros(T*N+1)
    for j in range(0, T*N+1): #build up the nodes from the bottom
      S[j] = S0 * u**j * d**(T*N-j)

  #Option payoff if exercising at all nodes
    C = np.zeros(T*N+1)
    for j in range(0, T*N+1):
      C[j] = max(0, S[j] - K)

  #Backward recursion through the tree: at each node, is it optimal to exercise or not?
    for i in np.arange(T*N-1,-1,-1):
      for j in range(0,i+1):
        S = S0 * u**j * d**(i-j)                      #S is function of j (#ups) and i-j (#downs)
        vested = (i+j >= v*N)

        if not vested:                                #Unvested
          C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
        elif vested & (S>=K*m):                       #Vested and early exercisable (as function of multiple - )
          C[j] = S - K
        elif vested & (S<K*m):                        #Vested but unexercisable (as function of multiple)
          C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )

    return C[0]


## COST OF R OPTION
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def r_eso_mod(alpha, gamma, S0=S_0, K=S_0, T=10, v=2, r=r, N=500, sigma=sigma_c, m=2):
      #Init values
    dt = 1/N                        #number of steps
    u = np.exp(sigma * np.sqrt(dt)) #using CRR method with (constant) volatility
    d = 1/u                         #to maintain the triangular structure of the tree (i.e., recombinant tree)
    q = (np.exp(r*dt) - d)/(u-d)    #q is the RN probability
    disc = np.exp(-r*dt)            #discount


  #Build up stock price tree (needed for next step)
    S = np.zeros(T*N+1)
    for j in range(0, T*N+1): #build up the nodes from the bottom
      S[j] = S0 * u**j * d**(T*N-j)

  #Option payoff if exercising at all nodes
    C = np.zeros(T*N+1)
    for j in range(0, T*N+1):
      C[j] = max(0, S[j] - K)

  #Backward recursion through the tree: at each node, is it optimal to exercise or not?
    for i in np.arange(T*N-1,-1,-1):
      for j in range(0,i+1):
        S = S0 * u**j * d**(i-j)                      #S is function of j (#ups) and i-j (#downs)
        vested = (i+j >= v*N)

        if not vested:                                #Unvested
          C[j] = disc * (q*C[j+1] + (1-q)*C[j])
        elif vested & (S >= K*m):                       #Vested and exercisable (as function of multiple)
          C[j] = (1+gamma)*(S - K) + (1-alpha+gamma)*(black_scholes(S,K,T,r,sigma)-(S-K))
        elif vested & (S < K*m):                        #Vested but unexercisable (as function of multiple)
          C[j] = disc * (q*C[j+1] + (1-q)*C[j])

        #NON-VESTED:
          #C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
          #C[j] = max(C[j], S - K)

    return C[0]


def cost_RN():
    return rn_eso()

def cost_R(alpha, gamma):
    return costs_r[(alpha, gamma)]

def run_costs():
    for alpha in alphas:
        for gamma in gammas:
            start_time = time.time()
            print(alpha, gamma)
            costs_r[(alpha, gamma)] = r_eso_mod(alpha, gamma)
            print("For R: --- %s seconds ---" % (time.time() - start_time))
    return costs_r


## FLAG CONSTRAINTS FOR THE DIFFERENT EQUILIBRIA ##
def check_constraints (row):
    #Check if IR is satisfied for both agents
    if row[4] >= U_hat: row[14] = 1
    if row[7] >= U_hat: row[15] = 1


    #Check if IC is satisfied for both agents -- REWRITE THISS!!
    IC_L, IC_H = 1, 1
    for a in [a_l, a_h]:
        for sigma in [sigma_l, sigma_h]:
            if (row[5] < utilities[(a, sigma, rho_L, row[2], row[0], row[1])]): IC_L = 0
            if (row[9] < utilities[(a, sigma, rho_H, row[6], row[0], row[1])]): IC_H = 0
    row[16], row[17] = IC_L, IC_H

    #Check if IC2 is satisfied for both agents for 3rd best (if the previous one is not, then this one is not either!)
    IC2_L, IC2_H = 1, 1
    for theta in thetas:
        for a in [a_l, a_h]:
            for sigma in [sigma_l, sigma_h]:
                if (row[5] < utilities[(a, sigma, rho_L, theta, row[0], row[1])]): IC2_L = 0
                if (row[9] < utilities[(a, sigma, rho_H, theta, row[0], row[1])]): IC2_H = 0
    row[18], row[19] = IC2_L, IC2_H

    return row


## LABEL IF AND WHAT TYPE OF EQUILIBRIUM, FOR 1ST, 2ND, 3RD ##
def label_equilibrium (row):
    row[20], row[21], row[22] = str(row[20]), str(row[21]), str(row[22])
    #print(row[14], row[15], row[16], row[17], row[18], row[19])
    print(type(row[14]), type(row[15]), type(row[16]), type(row[17]), type(row[18]), type(row[19]))


    if (row[14] == "1.0") and (row[15] == "1.0"): #IR satisfied for both agents --> both agents are active
        if (row[16] == "1.0") and (row[17] == "1.0"): #IC1 satisfied for both agents
            if (row[18] == "1.0") and (row[19] == "1.0"): #IC2 satisfied for both agents
                if (row[3] == row[7]) and (row[4] == row[8]): #Pooling (same a and sigma)
                    row[20], row[21], row[22] = "Yes (pooling)", "Yes (pooling)", "Yes (pooling)"
                else: #Screening
                    row[20], row[21], row[22] = "Yes (screening)", "Yes (screening)", "Yes (screening)"
            else: #IC2 NOT satisfied for at least one agent
                if (row[3] == row[7]) and (row[4] == row[8]): #Pooling
                    row[20], row[21], row[22] = "Yes (pooling)", "Yes (pooling)", "No"
                else: #Screening
                    row[20], row[21], row[22] = "Yes (screening)", "Yes (screening)", "No"
        else:
            if (row[3] == row[7]) and (row[4] == row[8]): #Pooling (same a and sigma)
                row[20], row[21], row[22] = "Yes (pooling)", "No", "No"
            else:
                row[20], row[21], row[22] = "Yes (screening)", "No", "No"
    elif ((row[14] == "1.0") and (row[15] == "0.0")): #IR satisfied for one agent only
        if row[16] == "1.0":
            if row[18]=="1.0":
                row[20], row[21], row[22] = "Yes (shutdown)", "Yes (shutdown)", "Yes (shutdown)"
            else:
                row[20], row[21], row[22] = "Yes (shutdown)", "Yes (shutdown)", "No"
        else:
            row[20], row[21], row[22] = "Yes (shutdown)", "No", "No"
    elif ((row[14] == "0.0") and (row[15] == "1.0")): #IR satisfied for one agent only
        if row[17] == "1.0":
            if row[19]=="1.0":
                row[20], row[21], row[22] = "Yes (shutdown)", "Yes (shutdown)", "Yes (shutdown)"
            else:
                row[20], row[21], row[22] = "Yes (shutdown)", "Yes (shutdown)", "No"
        else:
            row[20], row[21], row[22] = "Yes (shutdown)", "No", "No"   
    else: #IR NOT satisfied for both agents
        row[20], row[21], row[22] = "No", "No", "No"
        
    return row


def export_to_excel(arrays):
    writer = pd.ExcelWriter('/Users/davordjekic/Desktop/Bocconi/Thesis/thesis_tex/code/equilibria.xlsx', engine='xlsxwriter')
    for i in range(len(arrays)):
        pd.DataFrame(arrays[i]).to_excel(writer, sheet_name='Sheet__'+str(i)+'_best', index=False)
        print("Saved sheet: ", i)
    writer.close()
    
    
## COMPUTE PRINCIPAL'S OPTIMAL CHOICE FOR 1ST, 2ND, 3RD BEST ##
def principal_choice (_lambda=lamb, _beta=beta):
    #Initialize array with columns: alpha, gamma, rho_L__theta, rho_L__a, rho_L__sigma, rho_L__u, rho_H__theta, rho_H__a, rho_H__sigma, rho_H__u, Exp_S_T, C_RN, C_R, P_u, IR_L, IR_H, IC_L, IC_H, IC2_L, IC2_H, First_eq, Second_eq, Third_eq
    utils = np.zeros((len(thetas)**2 * len(alphas) * len(gammas) * 2**4, 20))
    labels_eq = np.empty((len(thetas)**2 * len(alphas) * len(gammas) * 2**4,3), dtype=str)
    #Merge the two datasets
    utilss = np.concatenate((utils, labels_eq), axis=1)
    #print("Shape: ", utils.shape)
    #print("Utils: ", utils)
 
    #run_sims()
    #run_utils()
    run_costs()
    #print(utilities)
    #print(utilities.keys())

    #Compute cost of RN option
    utils[:, 11] = cost_RN()

    row = 0

    for theta_low in thetas:
        for theta_high in thetas:
            for alpha in alphas:
                for gamma in gammas:
                    for a_H in [a_l, a_h]:
                        for a_L in [a_l, a_h]:
                            for sigma_H in [sigma_l, sigma_h]:
                                for sigma_L in [sigma_l, sigma_h]:
                                    #print(theta_low, theta_high, alpha, gamma)

                                    utils[row, 0] = alpha
                                    utils[row, 1] = gamma

                                    utils[row, 2] = theta_low
                                    utils[row, 6] = theta_high


                                    #Compute agents' choices - REPLACED TEMPORARILY
                                    #utils[row, 3], utils[row, 4], utils[row, 5] = agent_choice(alpha, gamma, rho_L, theta_low)
                                    #utils[row, 7], utils[row, 8], utils[row, 9] = agent_choice(alpha, gamma, rho_H, theta_high)
                                    utils[row, 3], utils[row, 4] = a_L, sigma_L
                                    utils[row, 7], utils[row, 8] = a_H, sigma_H
                                    utils[row, 5] = utilities[(a_L, sigma_L, rho_L, theta_low, alpha, gamma)]
                                    utils[row, 9] = utilities[(a_H, sigma_H, rho_H, theta_high, alpha, gamma)]


                                    #Compute mu (agents choosing the RN option)
                                    #print(_lambda)
                                    a, b= (utils[row, 2], utils[row, 6])
                                    #print(a, b)    
                                    mu = _lambda * a + (1 - _lambda) * b
                                    

                                    #Compute total effort
                                    #print(type(_lambda), type(utils[row, 3]), type(utils[row, 7]))
                                    a_tot = _lambda * utils[row, 3] + (1 - _lambda) * utils[row, 7]

                                    #Compute E[S_T]
                                    utils[row, 10] = exp_terminal_stock(S_0, a_tot)

                                    #Compute cost of R_{alpha, gamma}
                                    utils[row, 12] = cost_R(alpha, gamma)
                                    
                                    #Compute principal's utility
                                    utils[row, 13] = _beta * (utils[row, 10] - (mu * utils[row, 11]) + (1 - mu) * utils[row, 12])

                                    #Check ALL constraints
                                    utils[row,:] = check_constraints(utils[row,:])

                                    #Label the type of equilibrium
                                    utilss[row,0:20] = utils[row,:]
                                    utilss[row,:] = label_equilibrium(utilss[row,:])

                                    row += 1


    #Sort array by principal's utility
    utils_sorted = utilss[utilss[:, 13].argsort()[::-1]]

    #Impose restrictions on the array: progressive filtering incorporates idea of increasing restriction
    first_best, second_best, third_best = [], [], []

    #1st best:
    for row in utils_sorted:
        if row[20] != "No":
            first_best.append(row)

    #2nd best:
    for row in first_best:
        if row[21] != "No":
            second_best.append(row)
    
    #3rd best:
    for row in second_best:
        if row[22] != "No":
            third_best.append(row)

    export_to_excel([utils_sorted, first_best, second_best, third_best])

    return utils_sorted, first_best, second_best, third_best


zero, first, second, third = principal_choice(lamb, beta)

print("Lenghts: ", len(zero), len(first), len(second), len(third))
#print("Zero:", zero)
#print("First:", first)
#print("Second:", second)
#print("Third:", third)






#IMPORT utilities[...] from Excel


#COMPUTE beforehand the cost of R and RN options