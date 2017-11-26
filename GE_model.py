import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import importlib
import pandas as pd
import funs
importlib.reload(funs);

class GEmodel:
    
    def __init__(self):
        
        self.mu = 0.15 #unemployment benefits
        self.beta = 0.99
        self.beta_con = 0.0
        self.rho = 2.0 #log utility
        self.alpha = 0.36 #capital share in the CD prod function
        self.delta = 0.025 #depreciation rate
        self.lbar = 1.0/0.9 #nor quite sure... slides says 0.9 but...
       
        #Simulations
        self.simN = 10000
        self.simT = 2000
        self.sim_burn_in = 100
        
        #3.Grids
        #a. Households
        self.a_min = 0.0
        self.a_max = 100
        self.Na = 100
        self.Nu = 2
        
        #b. capital
        self.K_min = 30
        self.K_max = 60
        self.NK = 1
        
        #4 Technical
        self.update_forecast = 0.3
        self.tol_forecast = 1e-4
        self.tol_cfunc = 1e-6
        self.unemp = 0.1 #ss unemployment level
        self.trans_u = np.matrix([[0.6 , 0.4], [0.04445  ,0.95555]])


        
    #5 Transition matrix
    @staticmethod
    def R_func(par, K):
        L = (1-par.unemp)*par.lbar
        R = 1 + par.alpha*((K/L)^(par.alpha-1)) - par.delta
        return R
    @staticmethod
    def K_func(par, R):
        L = (1-par.unemp)*par.lbar
        K = ((R - 1 + par.delta)/(par.alpha*((1.0/L)**(par.alpha-1))))**(1/(par.alpha-1))
        return K
    @staticmethod
    def W_func(par, K):
        L = (1-par.unemp)*par.lbar
        W = (1 - par.alpha)*((K/L)**(par.alpha))
        return W
    @staticmethod
    def marg_util(par, c):
        c = np.array(c, dtype=float)
        u = c**(-par.rho) + par.beta_con
        return u
    @staticmethod
    def inv_marg_util(par, u):
        u = np.array(u, dtype = float)
        c = (u - par.beta_con)**(-1/par.rho)
        return c
    
    @staticmethod
    def indexFunc(par, i_u, i_uplus):
        index = i_u*par.Nu + i_uplus
        return index
    
    @staticmethod
    def pFunc(par, i_u, i_uplus):
        p = par.trans_u.item(par.indexFunc(par, i_u, i_uplus))
        return p
    
    @staticmethod
    def createGrids(par):
        g = {}
        g['a'] = funs.nonlinspace(10e-6,par.a_max,par.Na,1.1)
        g['a_mat'] = np.tile(g['a'], (1, par.Nu))
        g['m'] = funs.nonlinspace(10e-6,par.a_max,par.Na,1.1)
        g['m_mat'] = np.tile(g['m'], (1, par.Nu))
        g['K'] = 50.0
        g['K_mat'] = np.tile(g['K'], (par.Na, par.Nu))
        g['L_mat'] = np.tile(1-par.unemp, (par.Na, par.Nu))
        g['h_mat'] = np.tile((0.0, 1.0), (par.Na,1))
        
        sol = {}
        sol['c'] = np.full((par.Na+1, par.Nu), np.nan)
        for i_u in range(par.Nu):
            Gm = np.insert(g['m'], 0, 0,0)
            sol['c'][:, i_u] = Gm.ravel()
        
        return g, sol
    
    @staticmethod
    def incomeParams(par, R):
        d = {}
        K = par.K_func(par, R)
        W = par.W_func(par, K)
        y_plus = np.full((2), np.nan)
        y_plus[0] = par.mu*W        
        fac = par.unemp/(1-par.unemp)
        y_plus[1] = (par.lbar - fac*par.mu)*W
        
        d['R_eq'] = R #denne skal være input i min funktion
        d['K_eq'] = K
        d['W_eq'] = W
        d['y_plus'] = y_plus
        
        return d
        
        
        
        
    
    

#%% Check supply and demand
par = GEmodel()
K_pf_ss = ((1/par.beta-(1-par.delta))/par.alpha)**(1/(par.alpha-1))
np.random.seed(2017)
    
#%% FInd consumption function
def findConsumptionFunction(par, sol, grids, vals):
    diff_cfunc = np.inf
    it = 0
    while diff_cfunc > par.tol_cfunc:
        
        
        it += 1
        c_old = sol['c'].copy()
        avg_marg_plus = np.full((par.Na, par.Nu), 0.0)
        for i_uplus in range(par.Nu):
            
            c_plus_interp = RGI([np.insert(grids['m_mat'][:,i_uplus], 0, 0.0)], values=sol['c'][:, i_uplus], method='linear', bounds_error=False, fill_value=None)
            m_plus = vals['R_eq']*grids['a_mat']  + vals['y_plus'][i_uplus]
            c_plus = np.reshape(c_plus_interp(m_plus.ravel()),m_plus.shape )
            marg_u_plus = par.marg_util(par, c_plus)
            trans_mat = np.repeat(par.trans_u[:,i_uplus].ravel(), par.Na, axis = 0)
            avg_marg_plus = avg_marg_plus + par.beta * vals['R_eq'] * np.multiply(trans_mat, marg_u_plus)
            
        c = par.inv_marg_util(par, avg_marg_plus)
        m = grids['a_mat'] + c
        
        for i_u in range(par.Nu):
            Gm = np.insert(grids['m'], 0, 0.0)
            M = np.insert(m[:,i_u], 0, 0.0)
            C = np.insert(c[:,i_u], 0, 0.0)
            c_interp = RGI([M], values=C, method='linear', bounds_error=False, fill_value=None)
            sol['c'][:, i_u] = c_interp(Gm)
        
        diff_cfunc = max(abs(c_old.ravel()-sol['c'].ravel()))
        
        assert(it < 2000)
    return sol

def drawRandomNumbers(par):
    #1 Allocate space
    sim = {}
    sim['h'] = 1 * np.ones((par.simN, par.simT), dtype=float)
    
    #2 Initial values
    ini_u = 1 * np.ones((par.simN, 1), dtype=float)
    Index = np.random.uniform(size=par.simN) < par.unemp
    ini_u[Index] = 0
    
    #3 Random numbers
    ph = np.random.uniform(size = (par.simN, par.simT))
    
    # time loop
    
    for t in range(par.simT):
        #1 lagges values
        if t == 0:
            h_lag = ini_u
        else:
            h_lag = sim['h'][:, t-1]
        
        #2 find new h
        
        for i_u in range(par.Nu):
            I = h_lag == i_u
            J = ph[:, t] < par.pFunc(par, i_u, 0) #prob for unemploeyed next period
            IJ = np.all([I.ravel(), J], axis=0)
            sim['h'][IJ, t] = 0
    return sim


def simulate(par, sol, sim, vals, a_dist, grids):
    # IN: par, sol, sim, a_dist
    # Out: sim, a_dist
    # 1 predifened values
    sim['L'] = np.full((par.simT), ((1- par.unemp)*par.lbar))
    sim['K'] = np.full((par.simT), np.nan)
    sim['R'] = np.full((par.simT), vals['R_eq'])
    sim['W'] = np.full((par.simT), vals['W_eq'])
    #2 time loop
    for t in range(par.simT):
        
        #1 capital
        sim['K'][t] = np.mean(a_dist)
        
        for i_u in range(par.Nu):
            
            #a select on h
            Index = sim['h'][:, t] == i_u #unemployed
            
            #b. Generate income
            if i_u == 0:
                y = par.mu * sim['W'][t]
            else:
                fac = par.unemp/(1 - par.unemp)
                y = (par.lbar - fac*par.mu) * sim['W'][t]
        
            #c Cash-on-hand
            m = sim['R'][t]*a_dist[Index] + y
            
            #d consumption
            Gm = np.insert(grids['m'], 0, 0.0)
            c_interp = RGI([Gm], values = sol['c'][:,i_u], method='linear', bounds_error=False, fill_value=None)
            c = c_interp(m)
            
            #e End-of-peride assets
            a_dist[Index] = m - c
        
        #2 Checking for boundaries
        Index = a_dist < par.a_min
        a_dist[Index] = par.a_min
        Index = a_dist > par.a_max
        a_dist[Index] = par.a_max
    return sim, a_dist

def findEquilibrium(par, R, print_sol = 1):
    grids, sol = par.createGrids(par)
    vals = par.incomeParams(par, R)
    sol = findConsumptionFunction(par, sol, grids, vals)
    sim = drawRandomNumbers(par)
    a_initial = np.full((par.simN), 45.0)
    sim1, a_1 = simulate(par, sol, sim, vals, a_initial, grids)
    #plt.plot(sim['K'])
    K_demand = vals['K_eq']
    K_supply = sim['K'][-1]
    diff = (K_supply - K_demand)**2
    if print_sol == 1:
        print('Demand:', K_demand)
        print('Supply:', K_supply)
        print('Interest rate:', R)
        print('Difference:', diff)
    return diff, K_supply, K_demand, R#, par, sol, sim, vals


#%%
results = []
r_lower = 1.0097
r_upper = 1.0097025
steps = 7
R = np.linspace(r_lower, r_upper, steps)
for rate in R: 
    diff, K_supply, K_demand, r = findEquilibrium(par, rate)
    results.append([diff, K_supply, K_demand, r])
    print('***')

df = pd.DataFrame(results)
df.columns = ['diff', 'K_supply', 'K_demand', 'r']
I = np.argmin(df['diff'])
res_eq = df.iloc[I, :]
print('solution, beta_con ==', par.beta_con, ':')
print(res_eq)

#%% nu med conspicious consumption
par = GEmodel()
par.beta_con = -0.1
r_lower = 1.00805555556
r_upper = 1.00808641976
steps = 10
results = []
R = np.linspace(r_lower, r_upper, steps)
for rate in R: 
    diff, K_supply, K_demand, r = findEquilibrium(par, rate)
    results.append([diff, K_supply, K_demand, r, par.beta_con])
#    print('Solved in (sec.):', time.time()-tid)
    print('***')

df_con = pd.DataFrame(results)
df_con.columns = ['diff', 'K_supply', 'K_demand', 'r', 'beta_con']
I = np.argmin(df_con['diff'])
res_eq_con = df_con.iloc[I, :]
print('solution, beta_con ==', par.beta_con, ':')
print(res_eq_con)

#%% Supplt and demand functions
r_lower = 1.001
r_upper = 1.0095
steps = 12
par = GEmodel()
R = np.concatenate((np.linspace(r_lower, r_upper, steps), np.linspace(r_upper+1e-5, 1.0102, 5)))

# Baseline
results = []
for i in range(len(R)):
    rate = R[i]
    diff, K_supply, K_demand, r = findEquilibrium(par, rate, print_sol = 0)
    results.append([diff, K_supply, K_demand, r])
    print(str(i+1) + ' out of ' + str(len(R)))
df_lines = pd.DataFrame(results)
df_lines.columns = ['diff', 'K_supply', 'K_demand', 'r']


par = GEmodel()
par.beta_con = -0.1

results_con = []
for i in range(len(R)):
    rate = R[i]
    diff, K_supply, K_demand, r = findEquilibrium(par, rate, print_sol = 0)
    results_con.append([diff, K_supply, K_demand, r])
    print(str(i+1) + ' out of ' + str(len(R)))
df_lines_con = pd.DataFrame(results_con)
df_lines_con.columns = ['diff', 'K_supply', 'K_demand', 'r']

#%% Colors
col3 = "#89F5FF"
col1 = "#006282"
col2 = "#E6216C"
csfont = {'fontname':'Cambria'}
hfont = {'fontname':'Helvetica'}


#%% Figure
fig, ax = plt.subplots(1)
plt.rcParams["font.family"] = "calibri"
plt.rcParams['font.size'] = 12
ax.plot(df_lines.K_supply, df_lines.r, '.-', color = col1, label = "K, supply, public")
ax.plot(df_lines_con.K_supply, df_lines_con.r, '.-', color = col2, label = "K, supply, private")
ax.plot(df_lines.K_demand, df_lines.r, '-', color = col3, label = "K, demand")
#plt.plot(df_lines_con.K_demand, df_lines_con.r, '--', label = "K, demand, consp.")
ax.axhline(y = res_eq_con['r'], color = col2, ls = ':')
ax.axhline(y = res_eq['r'], color = col1, ls = ':')
ax.set_ylim(1.002,1.011)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
ax.legend()
ax.set_xlabel("Capital")
ax.set_ylabel('Interest rate')
ax.set_xlim(10,65)
plt.show()
