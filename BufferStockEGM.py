import numpy as np
import matplotlib.pyplot as plt
import importlib
import BFClass
importlib.reload(BFClass)
from BFClass import model
import funs
importlib.reload(funs)
import FigModule
importlib.reload(FigModule)
from FigModule import solver;
#%% With retirement

par = model()
par.T = 90-20
par.simlifecycle = 1
par.RT = par.T - 20 # Retirement age
par.sim_mini = 1.5
# Income profile
par.L[0:par.RT] = np.linspace(1,1/(par.G),par.RT)
par.L[par.RT] = 0.90
par.L[par.RT:] = par.L[par.RT:]/par.G
# Create grids for state variables
par.grid = par.create_grids(par)

#%%
solution = []
beta_n = []
params = []
for b in [-10.0, 0.0]:
    par.beta_n = b
    beta_n.append(b)
    solution.append(solver(par))
    params.append(par)

sim = []
for i in range(len(solution)):
    sim.append(FigModule.Simulation(params[i], solution[i]))
#%% Colors
col = [np.nan, np.nan, np.nan]
col[0] = "#006282"
col[1] = "#E6216C"
col[2] = "#89F5FF"

case = []
for i in range(len(solution)):
    if beta_n[i] < 0:
        case.append('Public')
    else:
        case.append('Private')

#%% Figure with lags
lags = [1, 2, 3, 30,50]
fig, ax = plt.subplots(1)
for j in range(len(lags)):
    l = lags[j]
    t = par.T - l
    if j == 0:
        for i in range(len(solution)):
            ax.plot(solution[i]['m'][t], solution[i]['c'][t], label = case[i], color = col[abs(1-i)])
    else:
        for i in range(len(solution)):
            ax.plot(solution[i]['m'][t], solution[i]['c'][t], color = col[i])
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_xlabel("Cash-in-hand")
ax.set_ylabel("Consumption")
ax.grid()
ax.legend()
plt.show()

#%% Creating means of population and 
means = FigModule.createMeans(sim, solution)
FigModule.lifecycleFigures(means, case, col)
FigModule.MPC_figure(solution, case, col)
FigModule.consumptionFigure(means, case, col)
FigModule.savingsFigure(means, case, col)
FigModule.singlePersonPlot(par, sim, case, col, public = 1)
FigModule.singlePersonPlot(par, sim, case, col, public = 0)