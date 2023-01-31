import numpy as np
from util.estimators import iv_2sls, ols
from util.utilities import get_lower_triangular
from scipy.optimize import minimize
import pandas as pd
import itertools


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

        # Initialize all demand parameter estimates
        self.beta = []  # K_1 + D x K_2 + K_3 (K_3 + 1)/2  (All param)
        self.beta_bar_hat = []  # K_1 x 1 (Linear param)
        self.beta_o_hat = []  # D x K_2 (Param on indiv. char) (Gamma in problem 2)
        self.beta_u_hat = []  # K_3 x K_3 (Random coefficients) (Gamma in problem 4)
        self.delta = []  # (T*J x 1)
        self.init_parameter_estimates()

        # Marginal cost estimates
        self.c = []

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
        Draw random coefficients from a standard normal distribution.
        """
        return self.estimopts['stream'].standard_normal((self.data.dims['K_3'], self.estimopts['num_sim']))

    def get_model_market_shares(self, delta, beta_u, mean=True):
        """
        Recover analytic market shares using the logit expression. This function works with aggregate data;
        see get_indiv_choice_prob for predicted choice probabilities for individuals.

        :param delta: (T x J) matrix of mean indirect utilities by product and market
        :param beta_u: (K_3 x K_3) lower triangular matrix of random coefficients
        :param mean: (Bool) return numerically integrated choice probabilities
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
        # indirect_cond_util = np.clip(indirect_cond_util, None, 30)

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

    def get_indiv_choice_prob(self, delta, beta_o, agg=False):
        """
        Get individual choie probabilities using microdata.

        :param delta: (T x J) matrix of mean indirect utilities by market and product
        :param beta_o: (D x J) matrix of coefficients on interactions between household and product characteristics
        :param agg: (Bool) Return aggregate choice probabilities
        :return agg_choice, indiv_choice: Return either aggregate or individual choice probabilities
        """
        # Mean indirect utility delta (reshape (T x J) -> (T x J x I))
        delta = np.reshape(np.repeat(delta, self.data.dims['I']),
                           (self.data.dims['T'], self.data.dims['J'], self.data.dims['I']))

        # Observed individual taste variation
        d_beta_o = np.matmul(self.data.d, beta_o)
        d_beta_o_x = np.matmul(self.data.x_2, np.transpose(d_beta_o))

        # Deviation from mean indirect utility
        mu = d_beta_o_x

        # Indirect conditional utility
        indirect_cond_util = delta + mu  # T x J x I

        # Find numerator and denominator
        numer = np.exp(indirect_cond_util)
        denom = np.nansum(numer, 1, keepdims=True)
        denom = np.repeat(denom, self.data.dims['J'], axis=1)
        # np.testing.assert_array_less(numer, denom)

        # Divide to get a T x J x I matrix
        indiv_choice = numer / denom
        indiv_choice = np.transpose(indiv_choice, (2, 1, 0))[:, :, 0]  # (T x J x I) --> (I x J x T)
        agg_choice = np.nanmean(indiv_choice, axis=0)

        # Return aggregate or individual choice probabilities
        if agg:
            return agg_choice
        else:
            return indiv_choice

    def get_delta(self, beta_u=[], beta_o=[], noisy=False):
        """
        Recover the mean indirect utilities as a function of non-linear parameters

        :param beta_u: (K_3 x K_3) matrix of random coefficients
        :param beta_o: (D x K_2) matrix of interactions between prod. and hh. characteristics
        :param noisy: Switch to show convergence status and number of iterations
        :return: (T x J) matrix of mean indirect utilities
        """
        diff = np.inf
        niter = 1
        delta = self.estimopts['delta_init']
        while diff > self.estimopts['delta_tol'] and niter < self.estimopts['delta_max_iter']:
            # print(f"Iter: {niter}")
            old_delta = delta
            if beta_o == []:
                delta = self.contraction_map(delta, beta_u=beta_u)
            else:
                delta = self.contraction_map(delta, beta_o=beta_o)

            diff = np.amax(abs(delta - old_delta))
            niter += 1

        if noisy:
            print(f"Converged with diff: {diff} and iterations: {niter}")
            print(delta[2, 3])

        return delta

    def contraction_map(self, delta, beta_u=[], beta_o=[]):
        """
        One iteration of the standard contraction map from BLP'95

        :param delta: (T x J) matrix of mean indirect utilities
        :param beta_u: (K_3 x K_3) matrix of random coefficients
        :param beta_o: (D x K_2) matrix of interactions between prod. and hh. characteristics
        :return: (T x J) matrix of mean indirect utilities
        """
        if beta_o == []:
            return delta + np.log(self.data.s) - np.log(self.get_model_market_shares(delta, beta_u))
        else:
            return delta + np.log(self.data.s) - np.log(self.get_indiv_choice_prob(delta, beta_o, agg=True))

    def get_moments(self, beta_u, W):
        """

        :param beta_u:
        :param W:
        :return:
        """
        # Recover linear parameters as a function of non-linear parameters via 2SLS
        delta = np.reshape(self.get_delta(beta_u), (self.data.dims['T'] * self.data.dims['J'], 1))
        X = np.reshape(self.data.x_1, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['K_1']))
        Z = np.reshape(self.data.z, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['Z']))
        exog_chars = np.reshape(self.data.x_1[:, :, 1], (self.data.dims['T'] * self.data.dims['J'], 1))
        Z_with_exog = np.append(Z, exog_chars, 1)
        beta_bar, X = iv_2sls(X, Z_with_exog, delta, include_constant=True, return_X=True)

        # Unobserved quality as a function of parameters
        xi = delta - X @ beta_bar

        # Quadratic form of xi and z
        G_hat = (np.transpose(xi) @ Z_with_exog)
        Q = G_hat @ W @ np.transpose(G_hat)

        return Q.flatten()[0]

    def get_likelihood(self, delta, beta_o):
        """
        Evalualte the negative of the log-likelihood function

        :param delta: (I x J) matrix of mean indirect utilities
        :param beta_o: (D x K_2) matrix of interactions between prod. and hh. characteristics
        :return: Negative of the log-likelihood
        """
        # Get individual choice probabilities
        prob_chosen_alt = np.zeros((self.data.dims['I'], 1))
        prob_all = self.get_indiv_choice_prob(delta, beta_o)

        # Get chosen alternative
        chosen_alt = self.data.micro_data[self.data.model_vars['c']]

        # Get the model predicted probability of the chosen alternative
        for i in range(self.data.dims['I']):
            prob_chosen_alt[i] = prob_all[i, chosen_alt[i] - 1]

        return -np.sum(np.log(prob_chosen_alt))

    def mle_objective(self, x, conc_out=False):

        """
        The objective function for the MLE optimization routine

        :param x: Vector of parameters. Depends on whether we concentrate out mean indirect utilities
        :param conc_out: Switch to concentrate out mean indirect utilities
        :return: Likelihood function evaluated at (delta, beta_o) unpacked from x
        """
        if conc_out:
            beta_o = np.reshape(x, (self.data.dims['D'], self.data.dims['K_2']))
            delta = self.get_delta(beta_o=beta_o)
        else:
            delta = x[:self.data.dims['J']]
            beta_o = np.reshape(x[self.data.dims['J']:], (self.data.dims['D'], self.data.dims['K_2']))
        return self.get_likelihood(delta, beta_o)

    # Should we also pass a weighting matrix?
    def blp_objective(self, beta_u, W):
        """
        Objective function for the BLP optimization routine

        :param beta_u: (K_3 x K_3) matrix of random coefficients
        :param W: (Z x Z) weighting matrix for GMM
        :return: scalar gmm objective
        """
        # Break up parameter vector
        beta_u = get_lower_triangular(beta_u)
        return self.get_moments(beta_u, W)

    def estimate(self):
        """
        Wrapper for estimation routines
        """
        # No need to do two-step since
        # we are not reporting std. errors
        if self.modeltype == "blp":
            self.estimate_blp()
        elif self.modeltype == "logit":
            if self.estimatortype == "gmm":
                self.estimate_blp()
            elif self.estimatortype == "2sls":
                self.estimate_logit()

            self.marginal_costs()
        else:
            self.estimate_micro(conc_out=True)

    def estimate_blp(self):
        """
        Estimate the BLP model as per the specifcations of Question 2
        """
        # GMM
        print("Estimating non-linear parameters...")
        beta_tilde_init = self.beta[self.data.dims['K_1']:]
        res = minimize(lambda x: self.blp_objective(x, np.eye(self.data.dims['Z'] + 1)), beta_tilde_init,
                       method='Nelder-Mead')
        self.beta_u_hat = get_lower_triangular(res.x[self.data.dims['D'] * self.data.dims['K_2']:])
        print(f"The estimates of random coefficients = {self.beta_u_hat}\n")

        # 2SLS
        print("Estimating linear parameters...")
        self.delta = np.reshape(self.get_delta(self.beta_u_hat),
                                (self.data.dims['T'] * self.data.dims['J'], 1))

        X = np.reshape(self.data.x_1, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['K_1']))
        Z = np.reshape(self.data.z, (self.data.dims['T'] * self.data.dims['J'], self.data.dims['Z']))
        exog_chars = np.reshape(self.data.x_1[:, :, 1], (self.data.dims['T'] * self.data.dims['J'], 1))
        Z_with_exog = np.append(Z, exog_chars, 1)
        self.beta_bar_hat = iv_2sls(X, Z_with_exog, self.delta, include_constant=True)[1:]
        print(f"The estimates of the linear parameters = {self.beta_bar_hat}\n")

        # Recover elasticities
        self.elasticities = self.compute_elasticities()
        print(f"Elasticity matrix = {self.elasticities}\n")

    def estimate_logit(self):
        """
        Estimate the logit model as per the specifications of Question 3
        """
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
        Z = np.concatenate((Z, X[:, 1:]), axis=1)

        # Recover parameters via OLS
        self.beta_bar_hat = iv_2sls(X, Z, log_shares_outside, include_constant=True)[1:]
        print(f"The estimates (logit) = {self.beta_bar_hat}\n")

        # Recover elasticities
        self.elasticities = self.compute_elasticities()
        print(f"Elasticity matrix = {self.elasticities}\n")

    def estimate_micro(self, conc_out=False):
        """
        Estimate the model in question 1
        :param conc_out: Concentrate out mean indirect utilities using the BLP contraction
        """
        print("Estimating mean indirect utilities and interactions by MLE...")

        # Recover MLE estimates of delta and Gamma
        if conc_out:
            x0 = self.beta_o_hat.flatten()
            res = minimize(lambda x: self.mle_objective(x, conc_out=True), x0, method='Nelder-Mead')
            self.beta_o_hat = np.reshape(res.x, (self.data.dims['D'], self.data.dims['K_2']))
            self.delta = np.transpose(self.get_delta(beta_o=self.beta_o_hat))
        else:
            x0 = np.concatenate((self.delta.flatten(), self.beta_o_hat.flatten())).ravel()
            res = minimize(self.mle_objective, x0, method='Nelder-Mead')
            self.delta = res.x[:self.data.dims['J']]
            self.beta_o_hat = np.reshape(res.x[self.data.dims['J']:], (self.data.dims['D'], self.data.dims['K_2']))

        print(res)
        print(f"The mean indirect utilities are given delta = {self.delta}\n")
        print(f"The interaction parameters are = {self.beta_o_hat}\n")

        # Recover OLS estimates of beta bar
        X = np.reshape(self.data.x_1, (self.data.dims['J'], self.data.dims['K_1']))
        self.beta_bar_hat = ols(X, self.delta, include_constant=True)[1:]
        print(f"The linear parameters are = {self.beta_bar_hat}\n")

    def compute_elasticities(self):
        """
        Recover matrix of own and cross price elasticities averaged by market

        :return: (JxJ) matrix of elasticities
        """
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
        """
        Recover marginal costs from demand estimates using the pricing euqation derived via the Bertrand pricing
        game.

        """
        # Initialize marginal cost array
        mc_array = np.zeros((self.data.dims['J'], 1))

        # Find average price, shares, and elasticities over all markets
        for j in range(self.data.dims['J']):
            average_pj = np.mean(self.data.x_1[:, j, 0], 0)
            average_sj = np.mean(self.data.s[:, j], 0)
            average_elast = self.compute_elasticities()[j, j]

            # Set the marginal cost equal to the correct formula.
            mc_array[j, 0] = average_pj + (average_sj / average_elast)

        # Set marginal cost vector
        print(f"Average marginal costs = {mc_array}\n")
        self.c = mc_array

    def counterfactuals_logit(self):
        # Set parameters
        one_over_alpha = -1 / self.beta_bar_hat[0]
        tolerance = 1e-6
        n_j = self.data.dims['J']
        shares_new = np.zeros((self.data.dims['T'] * n_j, 1))
        prices_new = np.reshape(self.data.x_1[:, :, 0], (self.data.dims['T'] * n_j, 1))
        numerator = np.zeros(n_j - 1)
        differences = np.zeros(n_j - 1)
        diff = np.repeat(np.inf, self.data.dims['T'])
        niter = 1

        # Make xi's
        x_1_beta_wide = np.matmul(self.data.x_1, self.beta_bar_hat)
        x_1_beta_long = np.reshape(x_1_beta_wide, (self.data.dims['T'] * n_j, 1))
        xi = self.delta - x_1_beta_long

        # Make a long version of x
        x_long = np.reshape(self.data.x_1[:, :, 1], (self.data.dims['T'] * n_j, 1))

        # Find profits and welfare before
        profits_pre = np.zeros(n_j)
        for i in range(n_j):
            profits_pre[i] = np.nanmean(self.data.s, 0)[i] * (np.nanmean(self.data.x_1[:, :, 0], 0)[i] - self.c[i, 0])

        welfare_pre = 0.577 - np.log(1 - sum(np.nanmean(self.data.s, 0)))

        # Loop over all 1000 markets
        for t in range(self.data.dims['T']):
            # print(t)
            while diff[t] > tolerance and niter < 10000:
                # print(niter)
                for i in range(n_j - 1):
                    # Find the numerator for all goods except good 1 (coded 0 in data)
                    numerator[i] = np.exp(
                        self.beta_bar_hat[0] * prices_new[n_j * t + i + 1, 0] + self.beta_bar_hat[1] * x_long[
                            n_j * t + i + 1, 0] + xi[n_j * t + i + 1, 0])
                for i in range(n_j - 1):
                    shares_new[n_j * t + i + 1] = numerator[i] / (sum(numerator) + 1)
                    newprice = self.c[i + 1, 0] + one_over_alpha + shares_new[n_j * t + i + 1] * (
                                prices_new[n_j * t + i + 1] - self.c[i + 1, 0])
                    differences[i] = abs(prices_new[n_j * t + i + 1] - newprice)
                    prices_new[n_j * t + i + 1] = newprice
                diff[t] = max(differences)
                niter += 1
        prices_new_reshape = np.reshape(prices_new, (self.data.dims['T'], n_j))
        shares_new_reshape = np.reshape(shares_new, (self.data.dims['T'], n_j))
        print(np.nanmean(prices_new_reshape, 0))
        print(np.nanmean(shares_new_reshape, 0))

        # Find profits and welfare after
        profits_post = np.zeros(n_j)
        for i in range(n_j):
            profits_post[i] = np.nanmean(shares_new_reshape, 0)[i] * (
                        np.nanmean(prices_new_reshape, 0)[i] - self.c[i, 0])

        welfare_post = 0.577 - np.log(1 - sum(np.nanmean(shares_new_reshape, 0)))

        # Change in profits and welfare
        profits_change = profits_post - profits_pre
        welfare_change = welfare_post - welfare_pre
        print(profits_change)
        print(welfare_change)

    def print_esimates(self, filename):
        pass

    def print_estimates(self, filename, title, label):
        """
        Print estimates from the model into a latex file

        :param filename: Path to file output
        :param title: Caption for table
        :param label: Label for table
        """
        # Get interaction coefficient names
        interactions = ["$" + i[0] + " \times " + i[1] + "$" for i in
                        itertools.product(self.data.model_vars['d'], self.data.model_vars['x_2'])]

        # Get all variable names
        var_names = ["\emph{Linear parameters}"]
        var_names.extend(self.data.model_vars['x_1'])
        var_names.extend(["\emph{Interact hh and prod. char}"])
        var_names.extend(interactions)
        var_names.extend(["\emph{Random coefficients}"])
        random_coeff_names = ["$\gamma_{%i,%i}$" % (i + 1, j + 1) for i, j in
                              itertools.product(range(self.data.dims['K_3']), range(self.data.dims['K_3']))]
        var_names.extend(random_coeff_names)

        # Get all data
        data = [" "]
        data.extend(["%.2f" % i for i in self.beta_bar_hat])
        data.extend([" "])
        data.extend(["%.2f" % i for i in list(self.beta_o_hat.flatten())])
        data.extend([" "])
        data.extend(["%.2f" % i for i in list(self.beta_u_hat.flatten())])

        # Make pandas dataframe
        data_dict = {}
        data_dict[self.estimatortype.upper()] = data
        df = pd.DataFrame(data=data_dict, index=var_names)
        print(df)

        # Print to location
        df.to_latex(buf=filename,
                    caption=title, label=label, index=True, escape=False)

    def print_elasticities(self, filename, title, label, format_float):
        """
        Print latex table of elasticities to disk.

        :param filename: Path to output
        :param title: Title for latex table
        :param label: Label for latex table
        :param format_float: Format strings to correctly format table entries
        """
        # Make dataframe out of elasticities
        df = pd.DataFrame(self.elasticities)
        df.to_latex(buf=filename, header=[str(j + 1) for j in range(self.data.dims['J'])],
                    float_format=format_float, caption=title, label=label, index=False)

    def print_averages(self, filename, title, label, format_float):

        if self.data.spec == "blp":
            x_1 = np.nanmean(self.data.x_1, axis=0)
            s = np.transpose(np.nanmean(self.data.s, axis=0))
            data = {"Price": x_1[:, 0], "Quality": x_1[:, 1], "Share": s}
            df = pd.DataFrame(data)
            df.to_latex(buf=filename, float_format=format_float, caption=title, label=label, index=False)
