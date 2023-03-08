import pandas as pd
import numpy as np
from scipy import sparse
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
        self.model = []          # Scitkit learn/keras model predicted bids
        self.homog = []          # Log homogenized bids
        self.mse = np.Inf        # Model prediction error

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
        y = self.bid_data['log_bid'].to_numpy()
        X = self.X.todense()[self.bid_data['id'],:]
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        if method == 'ols':
            print("Estimating gamma via least squares...")
            y = y[..., None]
            self.model = linear_model.LinearRegression().fit(X,y)
            yhat = self.model.predict(X)
            self.homog = y - yhat
            self.mse = metrics.mean_squared_error(y,yhat)
            print(f'Mean squared error (OLS): {self.mse:.2f}\n')
        elif method == 'lasso':
            print("Estimating gamma via lasso...")
            cv = 5
            self.model = linear_model.LassoCV(cv=cv,verbose=True, fit_intercept=True).fit(X_scaled,y)
            yhat = self.model.predict(X_scaled)
            self.homog = y - yhat
            self.mse = metrics.mean_squared_error(y,yhat)
            print(f'Number of cross-validation folds: {cv}')
            print(f'Penalty: {self.model.alpha_:.3f}')
            print(f'Mean squared error (Lasso): {self.mse:.2f}\n')

        elif method == 'neural':
            print("Estimating gamma via neural network...")
            self.model = neural_network.MLPRegressor()

        else:
            raise ValueError(f"Method not recognized: {method}")




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

