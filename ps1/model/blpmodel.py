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
        Initial the model object

        :param data: The Data object
        :param estimatortype: One of ("mle","gmm","2sls")
        :param estimopts: Optional vector of estimation options for GMM/MLE. Set to defaults if not provided
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
        Set default estimator options

        :return estimopts: Dictionary containing estimator options
        """
        delta_init = np.random.randn(self.data.dims["T"], self.data.dims["J"])
        estimopts = {
            'stream': np.random.default_rng(2023),
            'num_sim': 50,
            'delta_tol': 1e-12,
            'delta_max_iter': 10000,
            'jac_tol': 1e-8,
            'delta_init': delta_init
        }

        return estimopts

    def init_parameter_estimates(self):
        """
        Initialize parameter estimates using the estimopts dictionary.
        """

        # Linear parameters
        self.beta_bar_hat = self.estimopts['stream'].uniform(-1, 1, self.data.dims["K_1"])
        # print(self.beta_bar_hat)

        # Parameters on interactions of observed household chararacteristics and product characteristics
        self.beta_o_hat = self.estimopts['stream'].uniform(-1, 1, (self.data.dims["D"], self.data.dims["K_2"]))
        # print(self.beta_o_hat)

        # Random coefficients (gamma)
        self.beta_u_hat = np.tril(
            self.estimopts['stream'].uniform(0, 1, (self.data.dims["K_3"], self.data.dims["K_3"])))
        # print(self.beta_u_hat)

        # Full parameter vector
        self.beta = np.concatenate((self.beta_bar_hat, self.beta_o_hat.flatten(), self.beta_u_hat.flatten()))
        self.beta = self.beta[self.beta != 0]

        # Mean indirect utility
        self.delta = self.estimopts['delta_init']

    def draw_random_shocks(self):
        """

        """
        return self.estimopts['stream'].standard_normal((self.data.dims['K_3'], self.estimopts['num_sim']))

    def get_model_market_shares(self, delta, beta_u, mean=True):
        """

        :param delta: (T x J) matrix of mean indirect utilities by product and market
        :param beta_u: (K_3 x K_3) lower triangular matrix of random coefficients
        :param mean: return numerically integrated choice probabilities
        :return cond_choice : (T x J) matrix of predicted market shares for each good j in market t
        """

        # Mean indirect utility delta (reshape (T x J) -> (T x J x S))
        delta = np.reshape(np.repeat(delta, self.estimopts['num_sim']),
                           (self.data.dims['T'], self.data.dims['J'], self.estimopts['num_sim']))

        # Unobserved individual taste variation
        x3_beta_u_hat = np.matmul(self.data.x_3, beta_u)
        x3_beta_u_hat_nu = np.matmul(x3_beta_u_hat, self.nu)

        # Deviation from mean indirect utility
        mu = x3_beta_u_hat_nu

        # Indirect conditional utility
        indirect_cond_util = delta + mu  # T x J x S
        #indirect_cond_util = np.clip(indirect_cond_util, None, 30)

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(numer, 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], axis=1)
        # np.testing.assert_array_less(numer, denom)

        # Divide to get a T x J x S matrix
        cond_choice = numer / (1 + denom)

        # Take the mean over all S simulations if we want to take the mean
        if mean:
            cond_choice = np.nanmean(cond_choice, 2)  # T x J x S -> T x J

        return cond_choice

    def get_indiv_choice_prob(self, delta, beta_o):

        # Mean indirect utility delta (reshape (T x J) -> (T x J x I))
        delta = np.reshape(np.repeat(delta, self.data.dims['I']),
                           (self.data.dims['T'], self.data.dims['J'], self.data.dims['I']))

        # Observed individual taste variation
        d_beta_o = np.matmul(self.data.d, beta_o)
        d_beta_o_x = np.matmul(self.data.x_2, np.transpose(d_beta_o))

        # Deviation from mean indirect utility
        mu = d_beta_o_x

        # Indirect conditional utility
        indirect_cond_util = delta + mu  # T x J x S

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(numer, 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], axis=1)
        #np.testing.assert_array_less(numer, denom)

        # Divide to get a T x J x I matrix
        indiv_choice = numer / (1 + denom)

        return indiv_choice

    def get_delta(self, beta_u, noisy=False):

        diff = np.inf
        niter = 1
        delta = self.estimopts['delta_init']
        while diff > self.estimopts['delta_tol'] and niter < self.estimopts['delta_max_iter']:
            # print(f"Iter: {niter}")
            old_delta = delta
            delta = self.contraction_map(delta, beta_u)
            diff = np.amax(abs(delta - old_delta))
            niter += 1

        if noisy:
            print(f"Converged with diff: {diff} and iterations: {niter}")
            print(delta[2, 3])

        return delta

    def contraction_map(self, delta, beta_u):
        return delta + np.log(self.data.s) - np.log(self.get_model_market_shares(delta, beta_u))

    def get_moments(self, beta_u, W):

        # Recover linear parameters as a function of non-linear parameters via 2SLS
        delta = np.reshape(self.get_delta(beta_u), (self.data.dims['T']*self.data.dims['J'],1))
        X = np.reshape(self.data.x_1,(self.data.dims['T']*self.data.dims['J'],self.data.dims['K_1']))
        Z = np.reshape(self.data.z,(self.data.dims['T']*self.data.dims['J'],self.data.dims['Z']))
        exog_chars = np.reshape(self.data.x_1[:, :, 1], (self.data.dims['T']*self.data.dims['J'],1))
        Z_with_exog = np.append(Z, exog_chars, 1)
        beta_bar, X = iv_2sls(X, Z_with_exog, delta, include_constant=True, return_X=True)

        # Unobserved quality as a function of parameters
        xi = delta - X @ beta_bar

        # Quadratic form of xi and z
        G_hat = (np.transpose(xi) @ Z_with_exog)
        Q = G_hat @ W @ np.transpose(G_hat)

        return Q.flatten()[0]

    def get_likelihood(self, beta_bar, beta_o):
        pass

    # Should we also pass a weighting matrix?
    def blp_objective(self, beta_u, W):

        # Break up parameter vector
        beta_u = get_lower_triangular(beta_u)
        return self.get_moments(beta_u, W)

    def estimate(self):

        # No need to do two-step since
        # we are not reporting std. errors
        if self.modeltype == "blp":
            self.estimate_blp()
        elif self.modeltype == "logit":
            if self.estimatortype == "gmm":
                self.estimate_blp()
            elif self.estimatortype == "2sls":
                self.estimate_logit()
        else:
            pass

    def estimate_blp(self):

        # GMM
        print("Estimating non-linear parameters...")
        beta_tilde_init = self.beta[self.data.dims['K_1']:]
        res = minimize(lambda x: self.blp_objective(x, np.eye(self.data.dims['Z']+1)), beta_tilde_init,
                       method='Nelder-Mead',bounds=((0,10),(-1,1),(-1,1)))
        self.beta_u_hat = get_lower_triangular(res.x[self.data.dims['D'] * self.data.dims['K_2']:])
        print(f"The estimates of random coefficients = {self.beta_u_hat}")

        # 2SLS
        print("Estimating linear parameters...")
        self.delta = np.reshape(self.get_delta(self.beta_u_hat),
                                (self.data.dims['T'] * self.data.dims['J'], 1))

        X = np.reshape(self.data.x_1, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['K_1']))
        Z = np.reshape(self.data.z, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['Z']))
        exog_chars = np.reshape(self.data.x_1[:, :, 1], (self.data.dims['T'] * self.data.dims['J'], 1))
        Z_with_exog = np.append(Z, exog_chars, 1)
        self.beta_bar_hat = iv_2sls(X, Z_with_exog, self.delta, include_constant=True)[1:]
        print(f"The estimates of the linear parameters = {self.beta_bar_hat}")

    def estimate_logit(self):

        # Calculate the outside share by market
        outside_share = 1 - self.data.s.sum(axis=1, keepdims=True)

        # Compute log(s_j/s_0)
        share_by_outside_good = self.data.s / outside_share
        share_by_outside_good_long = np.reshape(share_by_outside_good, (6000, 1))
        log_shares_outside = np.log(share_by_outside_good_long)
        self.delta = log_shares_outside

        # Vectors of product characteristics and instruments
        X = np.reshape(self.data.x_1, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['K_1']))
        Z = np.reshape(self.data.z, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['Z']))
        Z = np.concatenate((Z, X[:,1:]),axis=1)

        self.beta_bar_hat = iv_2sls(X, Z, log_shares_outside, include_constant=True)[1:]
        print(f"The estimates (logit) = {self.beta_bar_hat}")


    def compute_elasticities(self):
        # Initialize the elasticities
        e = np.zeros((self.data.dims['J'], self.data.dims['J']))

        # Find the predicted market shares, with and without taking the mean
        s_hat = self.get_model_market_shares(self.delta, self.beta_u_hat, mean=False)
        s_hat_mean = self.get_model_market_shares(self.delta, self.beta_u_hat)

        # Loop over all values of J and then over all values of K
        for j in range(self.data.dims['J']):
            for k in range(self.data.dims['J']):
                # Break it out by case for if j = k or not
                if j != k:
                    # Find separately for j and k then multiply and get mean
                    s_hat_j = s_hat[:, j, :]
                    s_hat_k = s_hat[:, k, :]
                    j_times_k = np.multiply(s_hat_j, s_hat_k)
                    mean_j_times_k = np.nanmean(j_times_k, 1)

                    # Find alpha times p_k divided by predicted sj
                    alpha_p_k = np.multiply(self.beta_bar_hat[0], self.data.x_1[:, k, 0])
                    s_hat_mean_j = s_hat_mean[:, j]
                    scaling_factor = np.divide(alpha_p_k, s_hat_mean_j)

                    # Find full elasticities and average over all markets then output into matrix
                    full_elasticity = np.multiply(mean_j_times_k, scaling_factor)
                    average_elasticity = np.nanmean(full_elasticity, 0)
                    e[j, k] = -average_elasticity
                else:
                    # Find sj * (1 - sj)
                    s_hat_j = s_hat[:, j, :]
                    j_times_one_minus = np.multiply(s_hat_j, 1 - s_hat_j)
                    mean_j_times_one_minus = np.nanmean(j_times_one_minus, 1)

                    # Find negative alpha times p_j divided by sj
                    alpha_p_j = np.multiply(self.beta_bar_hat[0], self.data.x_1[:, j, 0])
                    s_hat_mean_j = s_hat_mean[:, j]
                    scaling_factor = np.divide(alpha_p_j, s_hat_mean_j)

                    # Find full elasticities and average over all markets then output into matrix
                    full_elasticity = np.multiply(mean_j_times_one_minus, scaling_factor)
                    average_elasticity = np.nanmean(full_elasticity, 0)
                    e[j, k] = average_elasticity
        return e

    # Function to back out marginal costs from the logit data (prices and estimated own-price elasticities)
    def marginal_costs(self):
        # Initialize marginal cost array
        mc_array = np.zeros((self.data.dims['J'], 1))

        # Find average price, shares, and elasticities over all markets
        for j in range(self.data.dims['J']):
            average_pj = np.mean(self.data.x_1[:, j, 0], 0)
            average_sj = np.mean(self.data.s[:, j], 0)
            average_elast = self.compute_elasticities()[j, j]

            # Set the marginal cost equal to the correct formula.
            mc_array[j, 0] = average_pj + (average_sj / average_elast)
        return mc_array

    def print_esimates(self, filename):
        pass

    def print_elasticities(self, filename):
        pass
