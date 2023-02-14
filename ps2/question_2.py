# coding=utf-8
import numpy as np
import pandas as pd
from util import flatten
import matplotlib.pyplot as plt


class Model:

    def __init__(self, filename):
        """
        Initialize the model object

        :param filename:
        """
        # Data
        self.data = pd.read_csv(filename)
        self.dims = self.get_data_dims()
        print(self.dims)

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
        self.alpha = 1
        self.beta = 2
        self.delta = 6
        self.gamma = 3
        self.rho = 0.8
        self.theta = [self.alpha, self.beta, self.delta, self.gamma, self.rho]

    def get_data_dims(self):
        """
        Return metadata dictionary.

        :return: Dictionary contain metadata
        """
        # Create data dictionary
        dims = {}
        dims['T'] = self.data['t'].nunique()
        dims['K'] = self.data['i'].nunique()
        dims['S'] = 100

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
        rng = np.random.default_rng(1950)

        # Set firm-market shocks
        self.eps_it = rng.normal(0, 1, (self.dims['K'], self.dims['T'], self.dims['S']))

        # Market level shocks
        self.eta_t = rng.normal(0, 1, (1, self.dims['T'], self.dims['S']))

    def compute_moments(self, theta):
        """
        Get the quadratic form of the empirical moments

        :param theta: List of parameters
        :return: Value of objective
        """
        G = (self.I_it - self.prob_entry(theta))  * self.z_it
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

    def get_objective(self,type='all'):
        """
        Compute the objective function as a partial function of one out of the five parameters

        :param type: String specifying the parameter which is allowed to vary; others fixed at true values.
        :return: The objective function as the function of the parameter allowed to vary.
        """
        if type == 'all':
            return self.compute_moments
        elif type == 'alpha':
            return lambda x: self.compute_moments(flatten([x, self.theta[1:]]))
        elif type == 'beta':
            return lambda x: self.compute_moments(flatten([self.theta[0], x, self.theta[2:]]))
        elif type == 'delta':
            return lambda x: self.compute_moments(flatten([self.theta[0:2], x, self.theta[3:]]))
        elif type == 'gamma':
            return lambda x: self.compute_moments(flatten([self.theta[0:3], x, self.theta[4]]))
        elif type == 'rho':
            return lambda x: self.compute_moments(flatten([self.theta[:4], x]))
        else:
            raise ValueError(f"Uknown type: {type}")

    def plot_objective(self, type, filename=""):

        values = {'alpha':np.linspace(-5,5,50),
                  'beta':np.linspace(-5,5,50),
                  'delta':np.linspace(0,12,100),
                  'gamma':np.linspace(0,5,50),
                  'rho':np.linspace(0,1,100)}
        objective = self.get_objective(type)
        x = values[type]
        y = [objective(t) for t in x]
        plt.plot(x,y)
        plt.ylabel("Objective")
        plt.xlabel(f"$\{type}$")
        if filename == "":
            plt.savefig(fname=f"figures/objective_param_{type}.pdf")
        else:
            plt.savefig(filename)
        plt.close()




def generate_data(T=100, filename="data/ps2q2.csv"):
    """
    Simulate entry data in T markets

    :param T: Number of markets
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
    generate_data(150, "data/ps2q2.csv")

    # Create model object
    model = Model("data/ps2q2.csv")
    model.plot_objective('alpha')
    model.plot_objective('beta')
    model.plot_objective('delta')
    model.plot_objective('gamma')
    model.plot_objective('rho')
