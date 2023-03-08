import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import linear_model
from sklearn import metrics


class BidModel:

    def __init__(self):

        # Data
        self.bid_data = []  # (3 x B) Pandas frame of bid characteristics
        self.num_pcpt = []  # (2 x A) Pandas frame of num. participants
        self.X = []  # Matrix of covariates

        # Homogenized bids
        self.model = []          # Scitkit learn/keras model predicted bids
        self.homog_log_bids = [] #

    def get_data(self):
        # Get the bids
        raw_bids = pd.read_csv('data/bids.csv')
        raw_bids['log_bid'] = np.log(raw_bids['bid_value'])
        self.bid_data = raw_bids

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

        if method == 'ols':
            self.model = linear_model.LinearRegression().fit(y,X)
        elif method == 'lasso':
            pass
        elif method == 'neural':
            pass
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
    model.homogenize_bids('ols')
