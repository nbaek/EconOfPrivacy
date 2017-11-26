import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RegularGridInterpolator
np.random.seed(2017)
def ConsumptionConvergence(par, sol):
    plt.figure()
    for t in [range(par.T-1, par.T - 3, -1), par.T - 10, par.T - 30, par.T - 50]:
        plt.plot(sol['m'][t], sol['c'][t], '-', label = "t = " + str(t))
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.grid()
    plt.legend()
    plt.show()
    return

def Simulation(par, sol):
    par.simT = par.T
    par.simN = 100000
    sim = {
            'm': np.full((par.simN, par.simT), np.nan),
            'c': np.full((par.simN, par.simT), np.nan),
            'a': np.full((par.simN, par.simT), np.nan),
            'p': np.full((par.simN, par.simT), np.nan),
            'y': np.full((par.simN, par.simT), np.nan)
            }
    
    
    #2 Shocks
    shocki = np.random.choice(par.grid['Nshocks'], size=(par.simN, par.simT), replace=True, p = par.grid['w'])
    sim['psi'] = par.grid['psi_vec'][shocki] #T
    sim['xi'] = par.grid['xi_vec'][shocki] #T
    pshock = 0.1
    
    #Make sure mean-1 shocks
    assert abs(sim['xi'].mean()-1)<1e-3
    assert abs(sim['psi'].mean()-1)<1e-3
    
    #3. Initial values
    sim['m'][:,0] = par.sim_mini #T
    sim['p'][:,0] = 0.0 #T
    
    #4. time loop simulation
    for t in range(par.simT):
        if par.simlifecycle == 0:
            c_interp = RegularGridInterpolator([sol['m'][0].ravel()], values=sol['c'][0].ravel(), method='linear', bounds_error=False, fill_value=None)
        else:
            c_interp = RegularGridInterpolator([sol['m'][t].ravel()], sol['c'][t].ravel(), method='linear', bounds_error=False, fill_value=None)
        
        sim['c'][:,t] = c_interp(sim['m'][:,t])
        sim['a'][:,t] = sim['m'][:,t] - sim['c'][:,t]
        if t < par.simT-1:
            if t+2 > par.RT:
                #pre retirement
                sim['m'][:, t+1] = (par.R * sim['a'][:, t]) / (par.G * par.L[t]) + 1
                sim['p'][:, t+1] = np.log(par.G) + np.log(par.L[t]) + sim['p'][:, t]
                sim['y'][:, t+1] = sim['p'][:, t+1]
            else:
                sim['m'][:, t+1] = par.R * sim['a'][:, t] / (par.G*par.L[t]*sim['psi'][:,t+1]) + sim['xi'][:,t+1]
                sim['p'][:, t+1] = np.log(par.G) + np.log(par.L[t]) + sim['p'][:, t] + np.log(sim['psi'][:, t+1])
                sim['y'][:, t+1] = sim['p'][:, t+1]   
        
        #5 renormalize
    sim['P'] = np.exp(sim['p'])
    sim['Y'] = np.exp(sim['y'])
    sim['M'] = sim['m']*sim['P']
    sim['C'] = sim['c']*sim['P']
    sim['A'] = sim['a']*sim['P']
    return sim

def solver(par):
    #Solver step
    tid = time.time()
    sol = {
            'm':[None]*par.T,
            'c':[None]*par.T
            }
    
    # Last periode - consume all
    sol['m'][par.T-1] = np.linspace(0, par.a_max, par.Na)
    sol['c'][par.T-1] = np.linspace(0, par.a_max, par.Na)
    
    #All other periods
    for t in range(par.T-2,-1,-1):
        c_plus_interp = RegularGridInterpolator([sol['m'][t+1]], sol['c'][t+1], method='linear', bounds_error=False, fill_value=None)
        #EGM steps
        sol['m'][t] = np.empty([par.Na, 1])
        sol['c'][t] = np.empty([par.Na, 1])
            
        for i_a in range(par.Na):
            a = par.grid['grid_a'][t][i_a]
            if (t+1) <= (par.RT-1): 
                fac = par.G*par.L[t]*par.grid['psi_vec']
                w = par.grid['w']
                xi = par.grid['xi_vec']
            else:
                fac = par.G*par.L[t]
                w = 1.0
                xi = 1.0
            
            inv_fac = 1/fac
            
            # b. future m and c (scalars)
            m_plus = inv_fac*par.R * a + xi
            c_plus = c_plus_interp(m_plus)
            m_plus = np.array(m_plus).ravel()
            c_plus = np.array(c_plus).ravel()
            fac = np.array(fac).ravel()
            w = np.array(w).ravel()
            # c. average future marginal utility (number)
            marg_u_plus = par.marg_utility(par, fac*c_plus)
            avg_marg_u_plus = np.sum(w*marg_u_plus)
            
            # d. current c
            sol['c'][t][i_a] = par.inv_marg_utility(par, par.beta*par.R*avg_marg_u_plus)
    
            # e. current m
            sol['m'][t][i_a] = a + sol['c'][t][i_a]
            
            #f add zero consumption
        sol['m'][t] = np.concatenate((par.grid['a_min'][t], sol['m'][t].ravel()))
        sol['c'][t] = np.concatenate((np.array([0.0]), sol['c'][t].ravel()))
    
    print("Solved in :", time.time() - tid)
    return sol

def MPC_figure(solution, case, col):
    fig, ax = plt.subplots()
    for i in range(len(solution)):
        MPC = np.diff(solution[i]['c'][0])/np.diff(solution[i]['m'][0])
        ax.plot(solution[i]['m'][0][:-1], MPC, label = case[i], color = col[abs(1-i)])
        print('MPC as m -> inf:', MPC[-1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlim(0,20)
    ax.set_xlabel("Cash-in-hand")
    ax.set_ylabel("Marginal consumption, MPC")
    ax.legend()
    plt.show()
    return

def lifecycleFigures(means, case, col, n_states=2):
    """
    Plots four aligning figures with assets, cash-in-hand, income and consumption.
    Plots are lifecycle plots.
    IN:
        means: list of means from simulation
        case: list of labels for private and public
        col: list of colors for lines.
        n_states: 2 if means contains solution for both private and public setting.
    OUT:
        One plot with four figures.
    """
    x = range(20,90)
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].set_title('Income')
    axarr[1, 0].set_title("Cash-in-hand")
    axarr[0, 1].set_title("Consumption")
    axarr[1, 1].set_title("End of period assets")
    for i in range(n_states):
        axarr[0, 0].plot(x, means['mY'][i], label = case[i], color = col[abs(1-i)])
        axarr[0, 0].legend()
        axarr[1, 0].plot(x, means['mM'][i], label = case[i], color = col[abs(1-i)])
        axarr[1, 0].legend()
        axarr[0, 1].plot(x, means['mC'][i], label = case[i], color = col[abs(1-i)])
        axarr[0, 1].legend()
        axarr[1, 1].plot(x, means['mA'][i], label = case[i], color = col[abs(1-i)])
        axarr[1, 1].legend()
    plt.show()
    return

def createMeans(sim, solution):
    """
    Calculation population means of income, assets, cash-in-hand and consumption.
    IN:
        sim: dictionary containing solution of simulation
        solution: dictionary containing the solution for the model.
    OUT:
        means: dictionary of pupulation means
    """
    means = {}
    means['mY'] = []
    means['mM'] = []
    means['mC'] = []
    means['mA'] = []
    
    for i in range(len(solution)):
        #Income
        means['mY'].append(np.mean(sim[i]['Y'], axis=0))
        means['mM'].append(np.mean(sim[i]['M'], axis=0))
        means['mC'].append(np.mean(sim[i]['C'], axis=0))
        means['mA'].append(np.mean(sim[i]['A'], axis=0))
    return means

def consumptionFigure(means, case, col, n_states = 2):
    """
    Plots lifecycle consumption for both private and public case
    IN:
        means: list of means from simulation
        case: list of labels for private and public
        col: list of colors for lines.
        n_states: 2 if means contains solution for both private and public setting.
    OUT:
        Plot of consumption
    """
    x = range(20,90)
    f, axarr = plt.subplots(1)
    #axarr.set_title('Consumption')
    for i in range(n_states):
        axarr.plot(x, means['mC'][i], label = case[i], color = col[abs(1-i)])
    axarr.spines['top'].set_visible(False)
    axarr.spines['right'].set_visible(False)
    axarr.spines['bottom'].set_visible(True)
    axarr.spines['left'].set_visible(True)
    axarr.set_xlabel("Age")
    axarr.set_ylabel("Consumption")
    axarr.axvline(x = 70, ls = ':', color = col[2])
    axarr.legend()
    plt.show()
    return

def savingsFigure(means, case, col, n_states=2):
    """
    Plots lifecycle savings (end-of-period-assets) for both private and public case
    IN:
        means: list of means from simulation
        case: list of labels for private and public
        col: list of colors for lines.
        n_states: 2 if means contains solution for both private and public setting.
    OUT:
        Plot of savings
    """
    x = range(20,90)
    f, axarr = plt.subplots(1)
    #axarr.set_title('Consumption')
    for i in range(n_states):
        axarr.plot(x, means['mA'][i], label = case[i], color = col[abs(1-i)])
    axarr.spines['top'].set_visible(False)
    axarr.spines['right'].set_visible(False)
    axarr.spines['bottom'].set_visible(True)
    axarr.spines['left'].set_visible(True)
    axarr.set_xlabel("Age")
    axarr.set_ylabel("Savings")
    axarr.axvline(x = 70, ls = ':', color = col[2])
    axarr.legend()
    plt.show()
    return
    
def singlePersonPlot(par, sim, case, col, public = 1):
    """
    Plotting a number of single agents from the simulation
    IN:
        par: model instance containing parameters
        sim: dictionary containing results of the simulation
        public: one of plotting agents under public regime
        case: list of labels
        col: list of color codes
    OUT:
        plots one figure with three plots woth cash-in-hand and consumption
    """
    x = range(90-par.simT,90)
    lemmings = [100, 3000, 10000]
    fig, ax = plt.subplots(len(lemmings), sharex=True)
    for i in range(len(lemmings)):
        lemming = int(lemmings[i])
        ax[i].plot(x, sim[public]['c'][lemming], '-.', label = "c", color = col[abs(1- public)])
        ax[i].plot(x, sim[public]['m'][lemming], '-', label = "m", color = col[abs(1- public)])
        ax[i].legend()
        ax[i].axvline(x = par.RT+20, ls = ":", color = col[abs(1- public)])
    plt.show()
    return
    