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
        delta_init = np.random.randn(self.data.dims["T"], self.data.dims["J"])
        delta_init[0] = 0
        estimopts = {
            'stream': np.random.default_rng(2023),
            'num_sim': 50,
            'delta_tol': 1e-12,
            'delta_max_iter': 100000,
            'jac_tol': 1e-8,
            'delta_init': delta_init
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
            self.estimopts['stream'].uniform(-0.05, 0.05, (self.data.dims["K_3"], self.data.dims["K_3"])))

        # Full parameter vector
        self.beta = np.concatenate((self.beta_bar_hat, self.beta_o_hat.flatten(), self.beta_u_hat.flatten()))
        self.beta = self.beta[self.beta != 0]

        # Mean indirect utility
        self.delta = self.estimopts['delta_init']

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
        # Exclude outside good
        assert(delta.shape[1] == self.data.dims['J']-1)

        if self.modeltype == "blp":
            # Mean indirect utility delta (reshape (T x J) -> (T x J x S))
            delta = np.resize(delta, (self.data.dims['T'], self.data.dims['J']-1, self.estimopts['num_sim']))

            # Observed individual taste variation
            d_beta_o = np.matmul(self.data.d, beta_o)
            d_beta_o_x = np.matmul(self.data.x_2, np.transpose(d_beta_o))

            # Unobserved individual taste variation
            x3_beta_u_hat = np.matmul(self.data.x_3, beta_u)
            x3_beta_u_hat_nu = np.matmul(x3_beta_u_hat, self.nu)

            # Deviation from mean indirect utility
            mu = x3_beta_u_hat_nu

            # Indirect conditional utility
            indirect_cond_util = delta + mu  # T x J x S
        else:
            indirect_cond_util = delta

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(numer, 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J']-1, axis=1)
        np.testing.assert_array_less(numer,denom)

        # Divide to get a T x J x S matrix
        cond_choice = numer /  (1+ denom)
        #print(np.amin(np.sum(cond_choice,axis=1) - 1))
        #print(np.max(np.sum(cond_choice,axis=1)-1))

        # Take the mean over all S simulations
        if self.modeltype == "blp":
            cond_choice = np.nanmean(cond_choice, 2)  # T x J x S -> T x J

        return cond_choice

    def get_delta(self, beta_o, beta_u):

        diff = np.inf
        niter = 1
        delta = self.estimopts['delta_init'][:,1:self.data.dims['J']]
        while diff > self.estimopts['delta_tol'] and niter < self.estimopts['delta_max_iter']:
            #print(f"Iter: {niter}")
            old_delta = delta
            delta = self.contraction_map(delta, beta_o, beta_u)
            diff = np.amax(abs(delta - old_delta))
            niter += 1

        print(f"Converged with diff: {diff} and iterations: {niter}")
        print(delta[2,2])
        return delta

    def contraction_map(self, delta, beta_o, beta_u):
        return delta + np.log(self.data.s[:,1:self.data.dims['J']]) - np.log(self.get_model_market_shares(delta, beta_o, beta_u))

    def get_moments(self, delta, beta_o, beta_u):
        pass

    def get_likelihood(self):
        pass

    # Should we also pass a weighting matrix?
    def objective(self, beta_o, beta_u):
        # Get delta estimate
        delta = self.get_delta(beta_o, beta_u)

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
