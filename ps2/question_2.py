# coding=utf-8
import numpy as np
import pandas as pd
from util import flatten
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import itertools


class Model:

    def __init__(self, filename="data/ps2q2.csv", seed=1950):
        """
        Initialize the model object

        :param filename: Path to data file
        """
        # Data
        self.data = pd.read_csv(filename)
        self.dims = self.get_data_dims()

        # Seed
        self.seed = seed

        # Data matrices
        self.x_t = []  # (1 x T) vector of market characteristics
        self.n_t = []  # (1 X T) vector of number of firms in market t
        self.z_it = []  # (K x T) matrix of firm-market characteristics
        self.I_it = []  # (K x T) matrix of indicators of whether firm i operates in market t
        self.get_data_matrices()

        # Structure of errors
        self.eps_it = []  # (K x T x S) tensor of idiosyncratic shocks
        self.eta_t = []  # (1 x T X S) tensor of market-level shocks
        self.draw_shocks()

        # Parameters (initialized to true values)
        self.theta = [1, 2, 6, 3, 0.8]

    def get_data_dims(self):
        """
        Return metadata dictionary.

        :return: Dictionary contain metadata
        """
        # Create data dictionary
        dims = {}
        dims['T'] = self.data['t'].nunique()
        dims['K'] = self.data['i'].nunique()
        dims['S'] = 150

        return dims

    def get_data_matrices(self):
        """
        Transform market and firm-market level data into appropriate matrices.
        """
        # Get market x firm level data
        self.z_it = np.reshape(self.data['z_it'].to_numpy(), (self.dims['K'], self.dims['T']), order='F')
        self.I_it = np.reshape(self.data['entered'].to_numpy(), (self.dims['K'], self.dims['T']), order='F')

        # Get market level data
        self.x_t = np.reshape(pd.unique(self.data['x_t']), (1, self.dims['T']))
        self.n_t = np.sum(self.I_it, axis=0)

    def draw_shocks(self):
        """
        Draw iid shocks for the simulation.
        """
        # Set seed
        rng = np.random.default_rng(self.seed)

        # Set firm-market shocks
        self.eps_it = rng.normal(0, 1, (self.dims['K'], self.dims['T'], self.dims['S']))

        # Market level shocks
        self.eta_t = rng.normal(0, 1, (1, self.dims['T'], self.dims['S']))

    def compute_moments(self, theta, interact='none'):
        """
        Get the quadratic form of the empirical moments

        :param theta: List of parameters
        :param interact: String describing the type of moment condition i.e. f(z_it,x_) = {1, z_it, z_it x_t}
        :return: Value of objective
        """
        if interact == 'none':
            G = (self.I_it - self.prob_entry(theta))
        elif interact == 'z':
            G = (self.I_it - self.prob_entry(theta)) * self.z_it
        elif interact == 'zx':
            G = (self.I_it - self.prob_entry(theta)) * self.z_it * self.x_t
        else:
            raise ValueError(f"Model type not recognized: {type}")
        mean_G = np.mean(G, axis=1)
        Q = np.matmul(np.transpose(mean_G), mean_G)

        return Q

    def prob_entry(self, theta):
        """
        Compute the model implied probability that firm i enters market t as a function of model parameters
        and data.

        :param theta: List of parameters
        :return: Pr{Iit(θ, xt, zt)|xt, zt} probabilities of entry (K x T)
        """
        # Compute idiosyncratic profits and sort within markets
        # from highest to lowest
        phi_it = self.phi_it(theta)
        keys = np.argsort(-phi_it, axis=0)
        phi_it = np.take_along_axis(phi_it, keys, axis=0)
        n = np.reshape(range(1, self.dims['K'] + 1), (self.dims['K'], 1))
        v = self.variable_profit(n, theta)
        pi_it = v[..., None] + phi_it

        # Find profitable firms
        profitable = pi_it > 0
        profitable = np.take_along_axis(profitable, np.argsort(keys, axis=0), axis=0)
        prob_entry = np.mean(profitable.astype(int), axis=2)

        return prob_entry

    def variable_profit(self, n, theta):
        """
        Return the variable component of profits v(x,n,theta) which is a function of the number of
        firms, market characteristics, and the parameters

        :param n: Number of firms in that market
        :param theta: List of parameters
        :return: variable profits (K x T)
        """
        return theta[3] + self.x_t * theta[1] - np.log(n) * theta[2]

    def phi_it(self, theta):
        """
        Return idiosyncratic and fixed component of profit (also referred to as 'profitability' in the text) and
        denoted φ_it. It is a function of the parameters, the errors, and the firm-market level characteristics.

        :param theta: List of parameters
        :return: fixed profits (K x T x S)
        """
        return self.z_it[..., None] * theta[0] + theta[4] * self.eta_t + np.sqrt(1 - theta[4] ** 2) * self.eps_it

    def get_objective(self, param='all', interact='none'):
        """
        Compute the objective function as a partial function of one out of the five parameters

        :param param: String specifying the parameter which is allowed to vary; others fixed at true values.
        :return: The objective function as the function of the parameter allowed to vary.
        """
        if param == 'all':
            return lambda x: self.compute_moments(x, interact=interact)
        elif param == 'alpha':
            return lambda x: self.compute_moments(flatten([x, self.theta[1:]]), interact=interact)
        elif param == 'beta':
            return lambda x: self.compute_moments(flatten([self.theta[0], x, self.theta[2:]]), interact=interact)
        elif param == 'delta':
            return lambda x: self.compute_moments(flatten([self.theta[0:2], x, self.theta[3:]]), interact=interact)
        elif param == 'gamma':
            return lambda x: self.compute_moments(flatten([self.theta[0:3], x, self.theta[4]]), interact=interact)
        elif param == 'rho':
            return lambda x: self.compute_moments(flatten([self.theta[:4], x]), interact=interact)
        else:
            raise ValueError(f"Unknown type: {param}")

    def plot_objective(self, param, interact=['none', 'z', 'zx'], filename=""):
        """
        Plot the objective function as a function of one of the parameters, holding the other
        parameters fixed at their true values.

        :param interact:
        :param param: String describing the parameter to let vary
        :param filename: Path to save figure
        """

        # Collect figure metadata
        values = {'alpha': (np.linspace(-2.5, 2.5, 50), 0),
                  'beta': (np.linspace(1, 3, 50), 1),
                  'delta': (np.linspace(4, 8, 100), 2),
                  'gamma': (np.linspace(2, 4, 50), 3),
                  'rho': (np.linspace(0, 1, 100), 4)}
        interact_map = {'none': ('$f(z_{it},x_t) = 1$', 'solid', 'black'),
                        'z': ('$f(z_{it},x_t) = z_{it}$', 'dotted', 'blue'),
                        'zx': ('$f(z_{it},x_t) = z_{it}x_t$', 'dashed', 'green')
                        }

        # Draw plots
        x = values[param][0]
        ymax = []
        ymin = []
        for i in interact:
            obj = self.get_objective(param=param, interact=i)
            y = [obj(val) for val in x]
            ymax.append(max(y))
            ymin.append(min(y))
            plt.plot(x, y, label=interact_map[i][0], linestyle=interact_map[i][1], color=interact_map[i][2])

        # Labels
        plt.ylabel("$\hat{Q}(\%s; \\theta_0)$" % param)
        plt.xlabel(f"$\{param}$")
        plt.legend()

        # Add vertical line at true value
        plt.axvline(x=self.theta[values[param][1]], color='red', linestyle='--')
        plt.text(x=self.theta[values[param][1]] + (max(x) - min(x)) / 20, y=(max(ymax) + min(ymin)) / 2,
                 s=f"$\{param}^* = {self.theta[values[param][1]]}$", color='red')

        # Save file
        if filename == "":
            plt.savefig(fname=f"figures/objective_param_{param}.pdf")
            print(f"Saving image to 'figures/objective_param_{param}.pdf'")
        else:
            plt.savefig(filename)
            print(f"Saving figure to {filename}")
        plt.close()

    def estimate(self, init, interact):
        """
        Estimate the parameters of the Berry (1992) model.

        :param interact:
        :param init: List of initial parameter guesses
        """
        # Recover the objective function
        print(f"Estimating parameters using the Nelder-Mead method with initial conditions = {init} and seed = {self.seed}")
        obj = self.get_objective(param='all', interact=interact)
        res = minimize(obj, x0=init, method='Nelder-Mead')
        self.theta = res.x


def hist_estimates(type='sim', interact='z'):
    """

    :param type:
    :param interact:
    """
    # Histogram of estimates based on different seed values but fixied initial guess
    est = []
    if type == 'sim':
        init = [1.5, 2.5, 7, 4, 0.5]
        seeds = range(1950, 2000)
        for seed in seeds:
            model = Model(seed=seed)
            model.estimate(init=init, interact=interact)
            est.append(model.theta)

    # Histogram of estimates based on different initial guesses but fixed seed
    elif type == 'init':
        alpha = flatten(np.linspace(-2.5, 2.5, 3).tolist())
        beta = flatten(np.linspace(1, 3, 3).tolist())
        delta = flatten(np.linspace(4, 8, 2).tolist())
        gamma = flatten(np.linspace(2, 5, 2).tolist())
        rho = flatten(np.linspace(0.7, 0.9, 2).tolist())
        init = itertools.product(alpha, beta, delta, gamma, rho)
        seed = 1950
        for val in init:
            model = Model(seed=seed)
            model.estimate(init=list(val), interact=interact)
            est.append(model.theta)
    else:
        raise ValueError(f'Interaction code not recognized: {interact}')

    # Prep data for histogram
    est = np.array(est)
    params = ['alpha', 'beta', 'delta', 'gamma', 'rho']
    for i in range(len(params)):
        x = flatten(est[:, i].tolist())
        plt.hist(x)
        plt.xlabel("$\hat{\%s}$" % params[i])
        plt.savefig(f'figures/hist_est_{params[i]}_{type}_{interact}.pdf')
        print(f"Saving figure to figures/hist_est_{params[i]}_{type}_{interact}.pdf'...")
        plt.close()


def generate_data(T=100, filename="data/ps2q2.csv"):
    """
    Simulate entry data in T markets

    :param T: Number of markets
    :param filename: Path to file
    :return: Pandas dataframe containing data
    """

    # Define function to recover equilibrium entrants
    def n_star(x_t, phi_it, theta):
        # Sort the idiosyncratic profitability within markets
        # from highest to lowest
        phi_it = np.reshape(phi_it, (T, K))
        x_t = np.reshape(x_t, (T, K))
        keys = np.argsort(-phi_it, axis=1)
        phi_it = np.take_along_axis(phi_it, keys, axis=1)
        n = np.reshape(range(1, K + 1), (1, K))

        # Find profitable firms
        pi_it = v(x_t, n, theta) + phi_it
        profitable = pi_it > 0
        n_star = np.sum(profitable, axis=1)

        # Reshape variables
        profitable = np.reshape(np.take_along_axis(profitable.astype(int), np.argsort(keys, axis=1), axis=1),
                                (T * K, 1)).flatten()
        n_star = np.reshape(np.repeat(n_star, K), (T * K, 1)).flatten()
        pi_it = np.reshape(np.take_along_axis(pi_it, np.argsort(keys, axis=1), axis=1), (T * K, 1)).flatten()

        return n_star, profitable, pi_it

    # Define function to recover variable profits as a function of number of entrants
    # and market characteristics.
    def v(x_t, n, theta):
        return theta['gamma'] + x_t * theta['beta'] - np.log(n) * theta['delta']

    # Set seed
    rng = np.random.default_rng(2023)

    # Number of potential entrants
    K = 30

    # Define parameters
    alpha = 1
    beta = 2
    delta = 6
    gamma = 3
    rho = 0.8
    theta = {
        'alpha': alpha,
        'beta': beta,
        'delta': delta,
        'gamma': gamma,
        'rho': rho
    }

    # Market characteristics
    x_t = rng.normal(0, 1, T)
    x_t = np.repeat(x_t, K)

    # Firm characteristics
    z_it = rng.normal(0, 2, T * K)

    # Shocks
    eta_t = rng.normal(0, 1, T)
    eta_t = np.repeat(eta_t, K)

    eps_it = rng.normal(0, 1, T * K)

    # Profits and entry
    phi_it = z_it * alpha + rho * eta_t + np.sqrt(1 - rho ** 2) * eps_it
    n_star, profitable, pi_it = n_star(x_t, phi_it, theta)

    # Construct pandas dataframe
    t = range(1, T + 1)
    t = np.repeat(t, K)
    i = range(1, K + 1)
    i = np.tile(i, T)
    df = {'t': t, 'i': i, 'x_t': x_t, 'z_it': z_it, 'num_firms': n_star, 'entered': profitable}
    df = pd.DataFrame(df)
    df.to_csv(filename, index=False)

    # Return data
    return df


if __name__ == '__main__':
    # Generate and store data
    generate_data(200)

    # Specify moment types
    interactions = ['none', 'z', 'zx']

    for i in interactions:
        # Create model object
        model = Model("data/ps2q2.csv")

        # Plot objective function
        model.plot_objective(param='alpha',interact=[i],filename=f'figures/fig_obj_param_alpha_{i}.pdf')
        model.plot_objective(param='beta', interact=[i], filename=f'figures/fig_obj_param_beta_{i}.pdf')
        model.plot_objective(param='delta', interact=[i], filename=f'figures/fig_obj_param_delta_{i}.pdf')
        model.plot_objective(param='gamma', interact=[i], filename=f'figures/fig_obj_param_gamma_{i}.pdf')
        model.plot_objective(param='rho', interact=[i], filename=f'figures/fig_obj_param_rho_{i}.pdf')

        # Estimate parameters
        hist_estimates(type='sim', interact=i)
        hist_estimates(type='init', interact=i)


