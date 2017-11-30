import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import RegularGridInterpolator

# Under udvikling
import importlib
import funs
importlib.reload(funs)

#
class model:
    """
    Class containing functions for the solving of 
    the DC-EGM model.
    Used for seminar Economics of Privacy at University of Copenhagen.
    """
    def __init__(self):
        """
        Initiates parametervalues for the model instance
        """
        #1 Demographical parameters
        self.T = 200
        self.RT = self.T
        #2 Agent preferences
        self.rho = 2
        self.beta = 0.96
        self.beta_n = 1.0
        
        #3 Income paramters
        ## 3.1 Standard variations
        self.sigma_psi = 0.1
        self.sigma_xi = 0.1
        
        # 3.2 Growth rate
        self.G = 1.03
        
        # 3.3 Low income shocks
        self.low_p = 0.005
        self.low_val = 0
        
        #4 Life-cycle
        self.L = np.ones([self.T])
        
        #5 Saving and borrowing
        self.R = 1.04
        self._lambda = 0.0
        
        #6  Numerical integration and grids         
        self.a_max = 20.0 # maximum point i grid for a
        self.a_phi = 1.1 # curvature parameters
        
        #7 number of elements
        self.Nxi  = 8 # number of quadrature points for xi
        self.Npsi = 8 # number of quadrature points for psi
        self.Na = 500 # number of points in grid for a
        
        #8 Simulation
        self.sim_mini = 2.5 # initial m in simulation
        self.simN = 500000 # number of persons in simulation
        self.simT = 100 # number of periods in simulation
        self.simlifecycle = 1 # 0 --> simulate infinite horizon model
    
    @staticmethod
    def create_grids(par):
        """
        Function that makes grid and shit.
        Input: class instance
        Output: dictionary with grids and stuff...
        """
        grid = {}
        
        #1 Define the model
        if par.sigma_xi == 0 and par.sigma_psi == 0 and par.low_p == 0:
            model = 'PF'
        else:
            model = 'BS'
        grid['Model'] = model
        
        # 2. Find shocks
        # 2.1 GaussHermite
        
        psi, psi_w = funs.GaussHermite_lognorm(par.sigma_psi, par.Npsi)
        xi, xi_w = funs.GaussHermite_lognorm(par.sigma_xi, par.Nxi)
        
        #2.3 Add low value income shock to xi
        if par.low_p > 0:
            xi = np.append(par.low_val, (xi - par.low_p * par.low_val)/(1-par.low_p))
            xi_w = np.append(par.low_p, (1 - par.low_p)*xi_w)
        
        psi_vec, xi_vec = np.meshgrid(psi, xi)
        psi_vec = np.matrix.flatten(psi_vec)
        xi_vec = np.matrix.flatten(xi_vec)

        psi_w_vec, xi_w_vec = np.meshgrid(psi_w, xi_w)
        psi_w_vec = np.matrix.flatten(psi_w_vec)
        xi_w_vec = np.matrix.flatten(xi_w_vec)
        
        w = psi_w_vec * xi_w_vec
        # Save the grids        
        grid['xi'] = xi
        grid['xi_vec'] = xi_vec
        grid['xi_w'] = xi_w
        grid['psi'] = psi        
        grid['psi_vec'] = psi_vec
        grid['psi_w'] = psi_w
        grid['psi_w_vec'] = psi_w_vec
        grid['xi_w_vec'] = xi_w_vec
        grid['w'] = w
        grid['Nshocks'] = len(w)
        
        if par._lambda == 0:
            a_min = np.zeros([par.T,1])
        else:
            psi_min = min(psi)
            xi_min = min(xi)
            a_min = np.array([np.nan for n in range(par.T)])
            for t in range(par.T-1,-1,-1):
                if t >= (par.RT-1):
                    Delta = 0 # No debt in last periode
                elif t == (par.T-2):
                    Delta = (par.R**(-1))*par.G*par.L[t+1]*psi_min*xi_min
                else:
                    Delta = (par.R**(-1))*(min(Delta,par._lambda) + xi_min)*par.G*par.L[t+1]*psi_min
                a_min[t] = -min(Delta, par._lambda)*par.G*par.L[t+1]*psi_min
        
        grid['a_min'] = a_min
        
        # End-of-period assets
        grid_a = [None]*par.T
        for t in range(par.T):
            grid_a[t] = funs.nonlinspace(a_min[t]+1e-6,par.a_max,par.Na,par.a_phi)

        grid['grid_a'] = grid_a
        AI = (par.R * par.beta)**(1/par.rho)
        cond = {
                'FHW':par.G/par.R,
                'AI':AI,
                'GI':AI*np.sum(w*psi_vec**(-1))/par.G,
                'RI':AI/par.R,
                'WRI': par.low_p**(1/par.rho) * (AI/par.R),
                'FVA': par.beta * np.sum(w*(par.G * psi_vec)**(1-par.rho))
                }
        grid['conditions'] = cond
        return grid
    
    @staticmethod        
    def utility(par, c):
        c = np.array(c, dtype=float)
        u = (c**(1-par.rho))/(1-par.rho) + par.beta_n * c
        return u
    @staticmethod
    def marg_utility(par, c):
        c = np.array(c, dtype=float)
        u = c**(-par.rho) + par.beta_n
        return u
    @staticmethod
    def inv_marg_utility(par, u):
        u = np.array(u, dtype=float)
        c = (u - par.beta_n)**(-1/par.rho)
        return c
    
    @staticmethod
    def solve(par):
        sol = {
                'm':[None]*par.T,
                'c':[None]*par.T
                }
        
        # Last periode - consume all
        sol['m'][par.T-1] = np.linspace(0, par.a_max, par.Na)
        sol['c'][par.T-1] = np.linspace(0, par.a_max, par.Na)
        
        #All other periods
        for t in range(par.T-2,-1,-1):
            c_plus_interp = RegularGridInterpolator([np.reshape(sol['m'][t+1],(par.Na,))], sol['c'][t+1], method='linear', bounds_error=False, fill_value=None)
            a = model.EGM_loop(par=par, sol=sol, t=t, c_plus_interp=c_plus_interp)
            sol['m'][t] = a['m'][t]
            sol['c'][t] = a['c'][t]
        return sol
    
    @staticmethod
    def EGM_vec(par, sol, t, c_plus_interp):
        """
        Function, that solves the model vectorized for efficiency.
        Input:
            t: integer. Time periode
            sol: solution dictionary for m and c
            c_plus_interp: Linear interpolation of next period consumption
        Output:
            returns sol - dictionary of m and c in current period
        """
        a_mat = np.tile(par.grid['grid_a'][t].T, (72, 1))
        if (t+1) <= (par.RT-1): # only relevant with absorbing state at time RT
            fac = par.G*par.L[t]*np.tile(np.reshape(par.grid['psi_vec'],(len(par.grid['psi_vec']),1)), (1, par.Na))
            w = np.tile(np.reshape(par.grid['w'], (len(par.grid['w']), 1)), (1, par.Na))
            xi = np.tile(np.reshape(par.grid['xi_vec'], (len(par.grid['w']), 1)), (1, par.Na))
            a = a_mat
#            print('***', t, 'Before retirement')
        else:
            fac = par.G*par.L[t]*np.ones((par.Na, 1)).T
            w = np.ones((par.Na, 1)).T
            xi = np.ones((par.Na, 1)).T
            a = par.grid['grid_a'][t].T
#            print('***', t, 'After retirement')
        
        inv_fac = 1/fac
        m_plus = inv_fac * par.R * a + xi
        c_plus = np.reshape(c_plus_interp(np.matrix.flatten(m_plus)), m_plus.shape)
        
        marg_util_plus = par.marg_utility(fac*c_plus)
        avg_marg_u_plus = np.sum(w*marg_util_plus, axis=0).T
        
        #Current c and m
        sol['c'][t] = np.reshape(par.inv_marg_utility(par.beta*par.R*avg_marg_u_plus), par.grid['grid_a'][t].shape)
        sol['m'][t] = par.grid['grid_a'][t] + sol['c'][t]
        return sol
    
    def EGM_loop(par, sol, t, c_plus_interp):
        """
        Function, that solves the model via looping over assets.
        Input:
            t: integer. Time periode
            sol: solution dictionary for m and c
            c_plus_interp: Linear interpolation of next period consumption
        Output:
            returns sol - dictionary of m and c in current period
        """
        
        sol['m'][t] = np.empty([par.Na, 1])
        sol['c'][t] = np.empty([par.Na, 1])
            
        for i_a in range(par.Na):
            a = par.grid['grid_a'][t][i_a]
            if (t+1) <= (par.RT-1): #kan være en nuance her
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
            marg_u_plus = par.marg_utility(fac*c_plus)
            avg_marg_u_plus = np.sum(w*marg_u_plus)
            
            # d. current c
            sol['c'][t][i_a] = par.inv_marg_utility(par.beta*par.R*avg_marg_u_plus)
    
            # e. current m
            sol['m'][t][i_a] = a + sol['c'][t][i_a]
            
            #f add zero consumption
        sol['m'][t] = np.concatenate((par.grid['a_min'][t], sol['m'][t].ravel()))
        sol['c'][t] = np.concatenate((np.array([0.0]), sol['c'][t].ravel()))
        
        return sol
    
    @staticmethod
    def simulation(par, sol):
        """
        Simulation of the model from the solved model.
        Input:
            par:= Parameters (class)
            sol:= Solution (dict)
        Output:
            Something something
        """
        #1 Setting up containers for solution
        

        return sim
        
        
        
        
        
        
        