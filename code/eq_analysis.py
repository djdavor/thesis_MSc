# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math


### 0 = RN option
### 1 = R option
simulations = {}

## INIT PARAMS ##
#### Brownian motion
points = 10000
paths = 100
mu_c = 0.0
sigma_c = 5.0 #0.3??
years = 20
S_0 = 30.0
r = 0.045

#### Principal
n_agents = 2 #10
beta = 1
delta = 1
lamb = 0.5
alphas = [0.5, 0.75, 1.0] #also more
gammas = [0, 0.1, 0.2] #also more
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
sigma_h = 0.1
sigma_l = 0
y_RN = 10
y_R1 = 10
y_R2 = 10



## SIMULATE BROWNIAN MOTION PATH WITH CONSTANT MEAN AND STANDARD DEVIATION -- NO NEED FOR SECOND BROWNIAN MOTION!!
def brownian_motion(S_0, a=0, sigma=0, is_RN = 0, is_R = 0): # is_RN = number of agents choosing RN option, is_R = number of agents choosing R option

    # Seed the random number generators
    rng = np.random.default_rng(42)
    rng_bis = np.random.default_rng(96) #Volatility's Brownian motion
    
    # Create the initial set of random normal draws
    Z = rng.normal(0.0, 1.0, (paths, points))
    Z_bis = rng_bis.normal(0.0, 1.0, (paths, points))

    # Define the time step size and t-axis
    interval = [0.0, 1.0]
    dt = (interval[1] - interval[0]) / (points - 1)
    t_axis = np.linspace(interval[0], interval[1], points)
    p_year = points / years

    # Use Equation 3.3 from [Glasserman, 2003] to sample brownian motion paths
    X = np.zeros((paths, points))
    X[:, 0] = S_0 # Set the initial value of the stock price
    for idx in range(points - 1):
        real_idx = idx + 1

        if (real_idx <= y_RN*p_year) and (real_idx <= (y_R1+y_R2)*p_year): #Both agents are exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + a * dt * (is_RN + is_R) + delta * sigma * np.sqrt(dt) * Z_bis[:, idx] + sigma_c * np.sqrt(dt) * Z[:, idx]
        
        elif (real_idx <= y_RN*p_year) and (real_idx > (y_R1+y_R2)*p_year): #Only RN agent is exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + a * dt * is_RN + delta * sigma * np.sqrt(dt) * Z_bis[:, idx] + sigma_c * np.sqrt(dt) * Z[:, idx]
            
        elif (real_idx > y_RN*p_year) and (real_idx <= (y_R1+y_R2)*p_year): #Only R agent is exerting effort
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt +  a * dt * is_R + delta * sigma * np.sqrt(dt) * Z_bis[:, idx] + sigma_c * np.sqrt(dt) * Z[:, idx]

        else: #No agent is exerting additional effort -- WHAT ABOUT VOLATILITY???
            X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + sigma * np.sqrt(dt) * Z_bis[:, idx] + sigma_c * np.sqrt(dt) * Z[:, idx]

    # Obtain the set of final path values
    final_values = pd.DataFrame({'final_values': X[:, -1]})

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

    # Output the mean and stdev of these final values
    #print(final_values.mean(), final_values.std())

    #Return the matrix of simulated paths
    return X 



def run_sims():
    for rn in [0, 1]:
        for r in [0, 1]:
            for a in [a_l, a_h]:
                for sigma in [sigma_l, sigma_h]:
                    simulations[(a, sigma, rn, r)] = brownian_motion(S_0, a, sigma, rn, r)
                

## COMPUTE AGENT'S UTILITY FUNCTION AS EXPECTATION OF INTEGRAL ## 
def agent_util(a, sigma, rho, theta, alpha, gamma):
    
    if theta == 0:
        sims = brownian_motion(S_0, a, sigma, 1, 0)
    else:
        sims = brownian_motion(S_0, a, sigma, 0, 1)

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


## COMPUTE EXPECTED TERMINAL STOCK PRICE ## -- NOTE THAT SIGMA PLAYS NO ROLE HERE (REFLECTS ALSO PRINCIPAL'S RN)
def exp_terminal_stock (S_0, a_tot):

    exp_value =  S_0 * np.exp(a_tot * years) #Formula for expected value of Geomtric Brownian Motion
    return exp_value


def agent_choice (alpha, gamma, rho, theta=1):

    util_a_h_sigma_h = agent_util(a_h, sigma_h, rho, theta, alpha, gamma)
    util_a_h_sigma_l = agent_util(a_h, sigma_l, rho, theta, alpha, gamma)
    util_a_l_sigma_h = agent_util(a_l, sigma_h, rho, theta, alpha, gamma)
    util_a_l_sigma_l = agent_util(a_l, sigma_l, rho, theta, alpha, gamma)

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
        

## COST OF RN OPTION ## ==> PUT THE CORRECT ONE!
def cost_RN():
    return 10

## COST OF R OPTION ==> PUT THE CORRECT ONE!
def cost_R (alpha, gamma):
    return 11*alpha + 10*gamma


## FLAG CONSTRAINTS FOR THE DIFFERENT EQUILIBRIA ##
def check_constraints (row):
    #Check if IR is satisfied for both agents
    if row[4] >= U_hat: row[14] = 1
    if row[7] >= U_hat: row[15] = 1


    #Check if IC is satisfied for both agents -- REWRITE THISS!!
    IC_L, IC_H = 1, 1
    for a in [a_l, a_h]:
        for sigma in [sigma_l, sigma_h]:
            if (row[5] < agent_util(a, sigma, rho_L, row[2], row[0], row[1])): IC_L = 0
            if (row[9] < agent_util(a, sigma, rho_H, row[5], row[0], row[1])): IC_H = 0
    row[16], row[17] = IC_L, IC_H

    #Check if IC2 is satisfied for both agents for 3rd best (if the previous one is not, then this one is not either!)
    IC2_L, IC2_H = 1, 1
    for theta in thetas:
        for a in [a_l, a_h]:
            for sigma in [sigma_l, sigma_h]:
                if (row[5] < agent_util(a, sigma, rho_L, theta, row[0], row[1])): IC2_L = 0
                if (row[9] < agent_util(a, sigma, rho_H, theta, row[0], row[1])): IC2_H = 0
    row[18], row[19] = IC2_L, IC2_H

    return row


def principal_choice (_lambda=lamb, _beta=beta):
    #Initialize array with columns: alpha, gamma, rho_L__theta, rho_L__a, rho_L__sigma, rho_L__u, rho_H__theta, rho_H__a, rho_H__sigma, rho_H__u, Exp_S_T, C_RN, C_R, P_u, IR_L, IR_H, IC_L, IC_H, IC2_L, IC2_H
    utils = np.zeros((len(thetas)**2 * len(alphas) * len(gammas), 20))
    run_sims()

    #Compute cost of RN option
    utils[:, 11] = cost_RN()

    row = 0

    for theta_low in thetas:
        for theta_high in thetas:
            for alpha in alphas:
                for gamma in gammas:
                    print(theta_low, theta_high, alpha, gamma)

                    utils[row, 0] = alpha
                    utils[row, 1] = gamma

                    utils[row, 2] = theta_low
                    utils[row, 6] = theta_high


                    #Compute agents' choices
                    utils[row, 3], utils[row, 4], utils[row, 5] = agent_choice(alpha, gamma, rho_L, theta_low)
                    utils[row, 7], utils[row, 8], utils[row, 9] = agent_choice(alpha, gamma, rho_H, theta_high)


                    #Compute mu (agents choosing the RN option)
                    mu = _lambda * utils[row, 2] + (1 - _lambda) * utils[row, 6]

                    #Compute total effort
                    a_tot = _lambda * utils[row, 3] + (1 - _lambda) * utils[row, 7]

                    #Compute E[S_T]
                    utils[row, 10] = exp_terminal_stock(S_0, a_tot)

                    #Compute cost of R_{alpha, gamma}
                    utils[row, 12] = cost_R(alpha, gamma)
                    
                    #Compute principal's utility
                    utils[row, 13] = _beta * (utils[row, 10] - (mu * utils[row, 11]) + (1 - mu) * utils[row, 12])

                    #Check ALL constraints
                    utils[row,:] = check_constraints(utils[row,:])

                    row += 1
    

    #Sort array by principal's utility
    utils_sorted = utils[utils[:, 13].argsort()[::-1]]

    #Impose restrictions on the array: progressive sorting incorporates idea of increasing restriction
    first_best, second_best, third_best = [], [], []

    #1st best:
    for row in utils_sorted:
        if (row[14] == 1) and (row[15] == 1):
            first_best.append(row)

    #2nd best:
    for row in first_best:
        if (row[16] == 1) and (row[17] == 1):
            second_best.append(row)
    
    #3rd best:
    for row in second_best:
        if (row[18] == 1) and (row[19] == 1):
            third_best.append(row)

    return utils_sorted, first_best, second_best, third_best

run_sims()
print(simulations)

#zero, first, second, third = principal_choice(lamb, beta)

print("Lenghts: ", len(zero), len(first), len(second), len(third))
print("Zero:", zero)
print("First:", first)
print("Second:", second)
print("Third:", third)





#Flag type equilibria
## Shutdown: 1 agent, 2 agents
## Pooling: Same a and same sigma
## Separating: else (either a or sigma different, both agents active)
 







## COMPARE WITH CONSTRAINED PROBLEM WITH ONLY RN OPTION ##

## note that y_rn = 10 (and similar) is the case if we had European options