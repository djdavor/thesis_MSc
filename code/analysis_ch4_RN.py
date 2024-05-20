from scipy.stats import norm
from matplotlib import pyplot as plt 
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress = True)

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

##PARAMETERS
# Option parameters
S0_init = 30  # S_0
K_init = 30  # Strike price (=S_0)
T_init = 10  # Maturity
v_init = 2  # Vesting period
r_init = 0.045  # RF rate
N_init = 50  # Height of tree
N_init_r = 50  # Height of tree for R option
sigma_init = 0.3  # Volatility

alpha_init = 0.75
gamma_init = 0
rho_init = 2


# Agent's parameters - 67/33
c_init = 1700000  # Agent's initial wealth
n_s_init = 110000  # Number of shares
n_o_init = 300  # Number of options



#COMPUTE UTILITY (W AS INPUT) AND INVERSE OF UTILITY (U AS INPUT)

#Compute power utility
def utility(w, rho): 
    if rho == 1:
        return np.log(w)
    else:
        return (w**(1-rho)) / (1-rho)


#Compute inverse of iso-elastic utility - needed to compute CE
def u_minus(u, rho):
    if rho==1:
        return np.exp(u)
    else:
        return (u*(1-rho))**(1/(1-rho))
    

#COMPUTE CE OF RN OPTION

def CE_rn(S0, K, T, v, r, N, sigma, rho, n_s, n_o, c):
    if n_o == 0:
        return 0

    # Init values
    dt = 1 / N  # number of steps
    u = np.exp(sigma * np.sqrt(dt))  # using CRR method with (constant) volatility
    d = 1 / u  # to maintain the triangular structure of the tree (i.e., recombinant tree)
    q_u = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) /
           (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
    q_d = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(r * dt / 2)) /
           (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2  # Probability of down move
    q_m = 1 - q_u - q_d

    # Build up stock price tree to compute exercise value
    S = np.zeros((T * N + 1, T * N + 1))
    U = np.zeros((T * N + 1, T * N + 1))

    for n in range(0, T * N + 1):
        for j in range(0, n + 1):
            S[j, n] = S0 * u ** j * d ** (n - j)  # j is #ups
            w = c * ((1 + (r / N)) ** n) + n_s * S[j, n] + n_o * np.maximum(0, S[j, n] - K)
            U[j, n] = utility(w, rho)  # Exercise value

    # Dynamic programming
    for i in np.arange(T * N - 1, -1, -1):  # we start from the lowest node one step before maturity
        S = S0 * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
        vested = (i >= v * N)

        cont_value = q_u * U[1:, i + 1] + q_d * U[:-1, i + 1] + q_m * U[:-1, i]
        excs_value = U[:-1, i]

        U[:-1, i] = np.where(vested, np.maximum(cont_value, excs_value), cont_value)

    E_c = (u_minus(U[0, 0], rho) - c - n_s * S0) / n_o

    return E_c


print(CE_rn(30, K_init, T_init, v_init, r_init, N_init_r, sigma_init, rho_init, n_s_init, n_o_init, c_init))

print(CE_rn(30, K_init, T_init, v_init, r_init, N_init_r, sigma_init, rho_init, n_s_init, n_o_init, c_init))

print(CE_rn(S0_init, K_init, T_init, v_init, r_init, N_init_r, sigma_init, rho_init, n_s_init, n_o_init, c_init))

print(CE_rn(30, K_init, T_init, v_init, r_init, N_init_r, sigma_init, rho_init, n_s_init, n_o_init, c_init))

print(CE_rn(30, K_init, T_init, v_init, r_init, N_init_r, sigma_init, rho_init, n_s_init, n_o_init, c_init))

