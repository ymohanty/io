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

        # Random coefficients
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

    def get_model_market_shares(self, delta, beta_o, beta_u):
        pass

    def get_delta(self, beta_o, beta_u, methods):
        pass

    def get_moments(self,delta, beta_o, beta_u):
        pass

    def get_likelihood(self):
        pass

    # Should we also pass a weighting matrix?
    def objective(self, beta_o, beta_u):
        # Get delta estimate
        delta = self.get_delta(beta_o, beta_u, "direct")

        # Get the moments
        G = self.get_moments(delta, beta_o, beta_u)

        G[np.where(np.isnan(G))] = 0
        mean_G = np.mean(G, 2)
        val_int = np.matmul(np.transpose(mean_G), W)
        val = np.matmul(val_int, mean_G)    # Value of objective

        # Inverse of covariance matrix
        inv_cov = np.linalg.inv(np.cov(np.transpose(G)))

    def estimate(self):
        pass

    def print_esimates(self, filename):
        pass

    def print_elasticities(self, filename):
        pass



