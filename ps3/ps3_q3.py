import pandas as pd
import numpy as np
from scipy import sparse
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import neural_network


class BidModel:

    def __init__(self):
        """

        """
        # Data
        self.bid_data = []  # (3 x B) Pandas frame of bid characteristics
        self.num_pcpt = []  # (2 x A) Pandas frame of num. participants
        self.X = []  # Matrix of covariates

        # Homogenized bids
        self.estimator = ""  # Estimator/model type
        self.model = []  # Scitkit learn/keras model predicted bids
        self.homog = []  # Log homogenized bids
        self.mse = np.Inf  # Model prediction error

        # Likelihood estimates
        self.params = []

    def get_data(self):
        """

        """
        # Get the bids
        self.bid_data = pd.read_csv('data/bids.csv')
        self.bid_data['log_bid'] = np.log(self.bid_data['bid_value'])
        self.bid_data['id'] = self.bid_data.groupby(['item_num']).ngroup()

        # Get the auction participants
        self.num_pcpt = pd.read_csv('data/items.csv')
        self.num_pcpt.rename(columns={'pred_n_participant': 'N'}, inplace=True)
        self.num_pcpt['N'] = self.num_pcpt['N'].astype(int)

        # Get item level attributes
        attr = pd.read_csv('data/sparse_attributes.csv')
        row = attr['i'].values - 1
        col = attr['j'].values - 1
        data = attr['fill']
        self.X = sparse.csr_matrix((data, (row, col)))

    def summarize_data(self):
        """

        """
        # Summary table
        bids_summary = pd.DataFrame(
            {
                'Mean': self.bid_data['log_bid'].mean(),
                'Std. dev': self.bid_data['log_bid'].std(),
                'Items': self.bid_data['item_num'].nunique(),
                'Bids': self.bid_data['log_bid'].count()
            },
            index=['Log-bids']
        )

        num_bidders_summary = pd.DataFrame(
            {
                'Mean': self.num_pcpt['N'].mean(),
                'Std. dev': self.num_pcpt['N'].std(),
                'Items': self.num_pcpt['N'].count(),
                'Bids': np.NAN
            },
            index=['Num. participants']
        )

        summary = pd.concat([bids_summary, num_bidders_summary])

        # Set the index to describe variable
        # summary.set_index(['Log-bids', 'Num. participants'])
        print(summary)
        print("\n\n")

        # Output
        summary.to_latex(
            'tables/tab_summary.tex',
            caption='Summary statistics for bids and participants',
            label='tab:sumStats',
            formatters={
                'Mean': "{:.2f}".format,
                'Std. dev': "{:.2f}".format,
                'Count': "{:n}".format,
                'Bids': "{:n}".format
            }
        )

    def homogenize_bids(self, method='ols'):
        """

        :param method:
        """

        self.estimator = method
        y = self.bid_data['log_bid'].to_numpy()
        X = self.X.todense()[self.bid_data['id'], :]
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        if method == 'ols':
            print("Estimating gamma via least squares...")
            y = y[..., None]
            self.model = linear_model.LinearRegression().fit(X, y)
            yhat = self.model.predict(X)
            self.homog = y - yhat
            self.mse = metrics.mean_squared_error(y, yhat)
            print(f'Mean squared error (OLS): {self.mse:.2f}\n')
        elif method == 'lasso':
            print("Estimating gamma via lasso...")
            cv = 5
            self.model = linear_model.LassoCV(cv=cv, verbose=True, fit_intercept=True).fit(X_scaled, y)
            yhat = self.model.predict(X_scaled)
            self.homog = y - yhat
            self.mse = metrics.mean_squared_error(y, yhat)
            print(f'Number of cross-validation folds: {cv}')
            print(f'Penalty: {self.model.alpha_:.3f}')
            print(f'Mean squared error (Lasso): {self.mse:.2f}\n')

        elif method == 'mlp':
            print("Estimating gamma via multi-layer perceptron...")
            self.model = neural_network.MLPRegressor(hidden_layer_sizes=(150, 100, 50), solver='adam',
                                                     max_iter=1000).fit(X_scaled, y)
            yhat = self.model.predict(X_scaled)
            self.homog = y - yhat
            self.mse = metrics.mean_squared_error(y, yhat)
            print(f'Mean squared error (MLP): {self.mse:.2f}\n')

        else:
            raise ValueError(f"Method not recognized: {method}")

        self.bid_data['homogenized'] = self.homog

    def summarize_homogenized_bids(self):
        """

        """

        # Plot histogram
        hist = sns.histplot(data=self.homog, kde=True)
        hist.set(xlabel="Log-homogenized bids (MLP)", ylabel="Count")

        # Add text
        mu = np.mean(self.homog)
        sigma = np.std(self.homog)
        plt.text(x=3, y=1200, s=f'$\mu: {mu:.3f}$')
        plt.text(x=3, y=1100, s=f'$\sigma: {sigma:.2f}$')
        plt.savefig(f'figures/hist_homog_bids_{self.estimator}.pdf')
        plt.close()

    def get_likelihood(self, param):

        # Get first and second highest bids
        y = self.bid_data.groupby('item_num').nth(0)['homogenized'].to_numpy()
        x = self.bid_data.groupby('item_num').nth(1)['homogenized'].to_numpy()

        # Get weights
        weights = self.get_gmm_weights()

        # Generate mixture model
        f = weights[0] * stats.norm.pdf(y, loc=param[0], scale=np.sqrt(param[1] ** 2)) \
            + weights[1] * stats.norm.pdf(y, loc=param[2], scale=np.sqrt(param[3] ** 2)) \
            + weights[2] * stats.norm.pdf(y, loc=param[4], scale=np.sqrt(param[5] ** 2))
        F = weights[0] * stats.norm.cdf(x, loc=param[0], scale=np.sqrt(param[1] ** 2)) \
            + weights[1] * stats.norm.cdf(x, loc=param[2], scale=np.sqrt(param[3] ** 2)) \
            + weights[2] * stats.norm.cdf(x, loc=param[4], scale=np.sqrt(param[5] ** 2))
        F[F > 0.9999999] = 0.9999999

        # Partial likelihood
        return -np.sum(np.log(f / (1 - F)))

    def estimate(self):

        # Recover estimates
        print("Estimating the parameters of the Gaussian-Mixture model by maximum likelihood...")
        init = [0, 1, 2, 4, 2, 1]
        res = minimize(lambda x: self.get_likelihood(x), x0=init,
                       bounds=(
                       (None, None), (0.0001, None), (None, None), (0.0001, None), (None, None), (0.0001, None)))
        self.params = res.x

        # Print table out
        df = pd.DataFrame(
            {
                "$\mu$":self.params[[0,2,4]],
                "$\sigma$":self.params[[1,3,5]]
            },
            index=["2-3 bidders","4-7 bidders","8+ bidders"]
        )

        print(df)

        df.to_latex(
            'tables/tab_param_estimates.tex',
            caption='MLE estimates of Gaussian-Mixture model',
            label='tab:mleParam',
            escape=False,
            formatters={
                "$\mu$":"{:.2f}".format,
                "$\sigma$":"{:.2f}".format
            }
        )

    def get_gmm_weights(self):

        # Define bins
        bin1 = [0, 1, 2, 3]
        bin2 = [4, 5, 6, 7]

        # Compute histogram
        weights = [0, 0, 0]
        N = self.num_pcpt['N'].to_list()
        for i in N:
            if i in bin1:
                weights[0] += 1
            elif i in bin2:
                weights[1] += 1
            else:
                weights[2] += 1

        # Normalize
        weights = np.array(weights) / len(N)
        return weights

    def plot_cdf(self):

        # Get data
        x = np.linspace(-100,100,1000)
        y1 = stats.norm.cdf(x, loc=self.params[0],scale=self.params[1])
        y2 = stats.norm.cdf(x, loc=self.params[2], scale=self.params[3])
        y3 = stats.norm.cdf(x, loc=self.params[4], scale=self.params[5])

        # Plot
        plt.plot(x,y1, color='r', label='2-3 bidders')
        plt.plot(x,y2, color='g', label='4-7 bidders')
        plt.plot(x,y3, color='b', label='8+ bidders')

        plt.xlabel('$v$')
        plt.ylabel('$F(v)$')
        plt.legend()
        plt.savefig('figures/fig_value_cdf.pdf')
        plt.close()


    def plot_optimal_reserve(self):
        pass


if __name__ == '__main__':
    # Part 1
    model = BidModel()
    model.get_data()
    model.summarize_data()

    # Part 2 -- See text

    # Part 3 -- See text

    # Part 4 -- See text

    # Part 5 -- See text

    # Part 6
    # (a)
    model.homogenize_bids('ols')

    # (b)
    model.homogenize_bids('lasso')

    # (b)
    model.homogenize_bids('mlp')

    # Part 7
    model.summarize_homogenized_bids()

    # Part 8
    model.estimate()
    model.plot_cdf()
