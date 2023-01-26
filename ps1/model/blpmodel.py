import numpy as np
import pandas as pd


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
        self.beta = []
        self.beta_bar_hat = []
        self.beta_o_hat = []
        # beta_u_hat is gamma in the context of this problem set
        self.beta_u_hat = []
        self.delta = []
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
        self.beta_u_hat = np.tril(self.estimopts['stream'].uniform(-1, 1, (self.data.dims["K_3"], self.data.dims["K_3"])))

        # Full parameter vector
        self.beta = np.concatenate((self.beta_bar_hat, self.beta_o_hat.flatten(), self.beta_u_hat.flatten()))
        self.beta = self.beta[self.beta != 0]

        # Mean indirect utility
        self.delta = np.zeros((self.data.dims["T"], self.data.dims["J"]))

    def draw_random_shocks(self):
        """

        """
        return self.estimopts['stream'].standard_normal((self.data.dims['K_3'], self.estimopts['num_sim']))

    def get_model_market_shares(self, delta, beta_o=None, beta_u=None):
        # First reshape delta to be T x J x S
        delta_reshape = np.resize(self.delta, (self.data.dims['T'], self.data.dims['J'], self.estimopts['num_sim']))

        # Get second term: x * Gamma * nu
        x3_betauhat = np.matmul(self.data.x_3, self.beta_u_hat)
        x3_betauhat_nu = np.matmul(x3_betauhat, self.nu)

        # Indirect conditional utility
        indirect_cond_util = delta_reshape + x3_betauhat_nu  # T x J x S

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(np.exp(indirect_cond_util[:, 1:self.data.dims['J']]), 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], 1)

        # Divide to get a T x J x S matrix
        cond_choice = numer/(1 + denom)

        # Take the mean over all S simulations
        cond_choice_mean = np.nanmean(cond_choice, 2)

        return cond_choice_mean

    def get_delta(self, beta_o, beta_u, methods):
        pass

    def get_moments(self,delta, beta_o, beta_u):
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



