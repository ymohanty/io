import numpy as np

# Function to compute 2SLS estimator
def iv_2sls(X, Z, Y):
    # Make transposes
    XT = np.transpose(X)
    ZT = np.transpose(Z)

    # Make estimator
    first_part = np.linalg.inv(XT @ Z @ np.linalg.inv(ZT @ Z) @ ZT @ X)
    second_part = XT @ Z @ np.linalg.inv(ZT @ Z) @ ZT @ Y
    estimator = np.matmul(first_part, second_part)

    return estimator
