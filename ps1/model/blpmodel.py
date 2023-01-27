import numpy as np
import pandas as pd
from util.estimators import iv_2sls
from util.utilities import get_lower_triangular
from scipy.optimize import minimize


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
        estimopts = {
            'stream': np.random.default_rng(2023),
            'num_sim': 150,
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
        #print(self.beta_bar_hat)

        # Parameters on interactions of observed household chararacteristics and product characteristics
        self.beta_o_hat = self.estimopts['stream'].uniform(-1, 1, (self.data.dims["D"], self.data.dims["K_2"]))
        #print(self.beta_o_hat)

        # Random coefficients (gamma)
        self.beta_u_hat = np.tril(
            self.estimopts['stream'].uniform(-0.05, 0.05, (self.data.dims["K_3"], self.data.dims["K_3"])))
        #print(self.beta_u_hat)

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
        :return cond_choice : (T x J) matrix of predicted market shares for each good j in market t
        """


        # Mean indirect utility delta (reshape (T x J) -> (T x J x S))
        delta = np.reshape(np.repeat(delta, self.estimopts['num_sim']),
                           (self.data.dims['T'],self.data.dims['J'], self.estimopts['num_sim']))

        # Observed individual taste variation
        d_beta_o = np.matmul(self.data.d, beta_o)
        d_beta_o_x = np.matmul(self.data.x_2, np.transpose(d_beta_o))

        # Unobserved individual taste variation
        x3_beta_u_hat = np.matmul(self.data.x_3, beta_u)
        x3_beta_u_hat_nu = np.matmul(x3_beta_u_hat, self.nu)

        # Deviation from mean indirect utility
        mu = x3_beta_u_hat_nu + d_beta_o_x

        # Indirect conditional utility
        indirect_cond_util = delta + mu  # T x J x S

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(numer, 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], axis=1)
        np.testing.assert_array_less(numer,denom)

        # Divide to get a T x J x S matrix
        cond_choice = numer /  (1+ denom)
        #print(cond_choice.shape)

        # Take the mean over all S simulations
        cond_choice = np.nanmean(cond_choice, 2)  # T x J x S -> T x J

        return cond_choice

    def get_delta(self, beta_o, beta_u, noisy=False):

        diff = np.inf
        niter = 1
        delta = self.estimopts['delta_init']
        while diff > self.estimopts['delta_tol'] and niter < self.estimopts['delta_max_iter']:
            #print(f"Iter: {niter}")
            old_delta = delta
            delta = self.contraction_map(delta, beta_o, beta_u)
            diff = np.amax(abs(delta - old_delta))
            niter += 1

        if noisy:
            print(f"Converged with diff: {diff} and iterations: {niter}")
            print(delta[2,3])

        return delta

    def contraction_map(self, delta, beta_o, beta_u):
        return delta + np.log(self.data.s) - np.log(self.get_model_market_shares(delta, beta_o, beta_u))

    def get_moments(self, beta_o, beta_u, W):

        # Recover linear parameters as a function of non-linear parameters via 2SLS
        delta = np.reshape(self.get_delta(beta_o, beta_u), (self.data.dims['T']*self.data.dims['J'],1))
        X = np.reshape(self.data.x_1,(self.data.dims['T']*self.data.dims['J'],self.data.dims['K_1']))
        Z = np.reshape(self.data.z,(self.data.dims['T']*self.data.dims['J'],self.data.dims['Z']))
        beta_bar = iv_2sls(X,Z,delta)

        # Unobserved quality as a function of parameters
        xi = delta - X @ beta_bar

        # Quadratic form of xi and z
        G_hat = (np.transpose(xi) @ Z)
        Q = G_hat @ W @ np.transpose(G_hat)

        return Q.flatten()[0]

    def get_likelihood(self, beta_bar, beta_o):
        pass

    # Should we also pass a weighting matrix?
    def blp_objective(self, beta_tilde, W):

        # Break up parameter vector
        beta_o = np.reshape(beta_tilde[:self.data.dims['D']*self.data.dims['K_2']],(self.data.dims['D'],self.data.dims['K_2']))
        beta_u = get_lower_triangular(beta_tilde[self.data.dims['D']*self.data.dims['K_2']:])
        return self.get_moments(beta_o, beta_u, W)

    def estimate(self):

        # No need to do two-step since
        # we are not reporting std. errors
        if self.modeltype == "blp":

            # GMM
            print("Estimating non-linear parameters...")
            beta_tilde_init = self.beta[self.data.dims['K_1']:]
            res = minimize(lambda x: self.blp_objective(x, np.eye(self.data.dims['Z'])), beta_tilde_init, method='Nelder-Mead')
            self.beta_o_hat = np.reshape(res.x[:self.data.dims['D']*self.data.dims['K_2']],(self.data.dims['D'],self.data.dims['K_2']))
            self.beta_u_hat = get_lower_triangular(res.x[self.data.dims['D']*self.data.dims['K_2']:])
            print(f"The estimates of the coefficients on observed chars = {self.beta_o_hat}")
            print(f"The estimates of random coefficients = {self.beta_u_hat}")

            # 2SLS
            print("Estimating linear parameters...")
            self.delta = np.reshape(self.get_delta(self.beta_o_hat,self.beta_u_hat), (self.data.dims['T']*self.data.dims['J'],1))
            X = np.reshape(self.data.x_1, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['K_1']))
            Z = np.reshape(self.data.z, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['Z']))
            self.beta_bar_hat = iv_2sls(X, Z, self.delta)
            print(f"The estimates of the linear parameters = {self.beta_bar_hat}")


        elif self.modeltype == "logit":
            pass
        else:
            pass

    def compute_elasticities(self):
        pass

    def print_esimates(self, filename):
        pass

    def print_elasticities(self, filename):
        pass
