import numpy as np
import pandas as pd


# from util import


class Model:

    # Set up an instance of this class
    def __init__(self, data, estimatortype, estimopts={}):
        """

        :param data:
        :param estimatortype:
        :param estimopts:
        """
        # Assign data object
        self.data = data

        # Get modeltype (one of "blp","logit","micro")
        self.modeltype = self.data.spec

        # Get estimatortype ("mle","gmm")
        self.estimatortype = estimatortype

        # Get estimator options
        if estimopts == {}:
            self.estimopts = self.get_estim_opts()
        else:
            self.estimopts = estimopts

        # Random taste shocks
        self.nu = self.draw_random_shocks()

        # Initialize all parameter estimates
        self.beta = []  # K_1 + D x K_2 + K_3 (K_3 + 1)/2  (All param)
        self.beta_bar_hat = []  # K_1 x 1 (Linear param)
        self.beta_o_hat = []  # D x K_2 (Param on indiv. char) (Gamma in problem 2)
        self.beta_u_hat = []  # K_3 x K_3 (Random coefficients) (Gamma in problem 4)
        self.delta = []  # Mean indirect utilities
        self.init_parameter_estimates()

        # initialize the elasticity matrix
        self.elasticities = []

    def get_estim_opts(self):
        """

        :return:
        """

        estimopts = {
            'stream': np.random.default_rng(2023),
            'num_sim': 100,
            'delta_tol': 1e-12,
            'delta_max_iter': 1000,
            'jac_tol': 1e-8,
            'delta_init': np.zeros((self.data.dims['T'], self.data.dims['J']))
        }

        return estimopts

    def init_parameter_estimates(self):
        """

        """

        # Linear parameters
        self.beta_bar_hat = self.estimopts['stream'].uniform(-1, 1, self.data.dims["K_1"])

        # Parameters on interactions of observed household chararacteristics and product characteristics
        self.beta_o_hat = self.estimopts['stream'].uniform(-1, 1, (self.data.dims["D"], self.data.dims["K_2"]))

        # Random coefficients (gamma)
        self.beta_u_hat = np.tril(
            self.estimopts['stream'].uniform(-1, 1, (self.data.dims["K_3"], self.data.dims["K_3"])))

        # Full parameter vector
        self.beta = np.concatenate((self.beta_bar_hat, self.beta_o_hat.flatten(), self.beta_u_hat.flatten()))
        self.beta = self.beta[self.beta != 0]

        # Mean indirect utility
        self.delta = np.zeros((self.data.dims["T"], self.data.dims["J"]))

    def draw_random_shocks(self):
        """

        """
        return self.estimopts['stream'].standard_normal((self.data.dims['K_3'], self.estimopts['num_sim']))

    def get_model_market_shares(self, delta, beta_o, beta_u):
        """

        :param delta: (T x J) matrix of mean indirect utilities by product and market
        :param beta_o: (D x K_2) matrix of coefficients on household/individual characteristics
        :param beta_u: (K_3 x K_3) lower triangular matrix of random coefficients
        :return: cond_choice_mean (T x J) matrix of predicted market shares for each good j in market t
        """
        # Mean indirect utility delta (reshape (T x J) -> (T x J x S))
        delta = np.resize(delta, (self.data.dims['T'], self.data.dims['J'], self.estimopts['num_sim']))

        # Observed individual taste variation
        d_beta_o = np.matmul(self.data.d, beta_o)
        d_beta_o_x = np.matmul(self.data.x_2, np.transpose(d_beta_o))

        # Unobserved individual taste variation
        x3_beta_u_hat = np.matmul(self.data.x_3, beta_u)
        x3_beta_u_hat_nu = np.matmul(x3_beta_u_hat, self.nu)

        # Deviation from mean indirect utility
        mu = d_beta_o_x + x3_beta_u_hat_nu

        # Indirect conditional utility
        indirect_cond_util = delta + mu  # T x J x S

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        print(np.amax(numer))
        print(np.amin(numer))
        denom = np.nansum(np.exp(indirect_cond_util[:, 1:self.data.dims['J']]), 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], 1)
        print(denom.shape)
        np.testing.assert_array_less(numer,1+denom)

        # Divide to get a T x J x S matrix
        cond_choice = numer / (1 + denom)

        # Take the mean over all S simulations
        cond_choice_mean = np.nanmean(cond_choice, 2)  # T x J x S -> T x J

        return cond_choice_mean

    def get_delta(self, beta_o, beta_u):

        diff = np.inf
        niter = 1
        delta = self.estimopts['delta_init']
        while diff > self.estimopts['delta_tol'] and niter < self.estimopts['delta_max_iter']:
            old_delta = delta
            delta = self.contraction_map(old_delta, beta_o, beta_u)
            diff = np.max(abs(delta - old_delta))
            #print(diff)
            niter += 1

        return delta

    def contraction_map(self, delta, beta_o, beta_u):
        return delta + np.log(self.data.s) - np.log(self.get_model_market_shares(delta, beta_o, beta_u))

    def get_moments(self, delta, beta_o, beta_u):
        pass

    def get_likelihood(self):
        pass

    def objective(self, beta_o, beta_u):
        pass

    def estimate(self):
        pass

    def print_esimates(self, filename):
        pass

    def print_elasticities(self, filename):
        pass
