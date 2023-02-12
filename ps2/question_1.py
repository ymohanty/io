import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import data

# Function to compute the objective function
def log_likelihood(params, n, char):
    """
    params: 3-dimensional array (phi, delta, beta)
    n: Number of firms
    char: Characteristic of market (x_t in problem set)
    Returns negative log likelihood of data given the parameters
    """
    return -sum(np.log(norm.cdf(params[0] + params[1]*np.log(n[i] + 1) - params[2]*char[i])
               - norm.cdf(params[0] + params[1]*np.log(n[i]) - params[2]*char[i]))
               for i in range(n.size))

def main():
    # Read in data and convert to arrays
    raw_data = pd.read_csv(data.DATA_LOC)
    data_array = raw_data.to_numpy()

    # Extract each column of data into separate arrays
    char_data = data_array[:, 0]
    n_data = data_array[:, 1]

    # Now minimize the negative log likelihood
    x0 = np.array([1, 1, 1])
    res = minimize(log_likelihood, x0, args=(n_data, char_data), method = 'Nelder-Mead')
    print(res)

if __name__ == '__main__':
    main()