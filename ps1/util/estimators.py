import numpy as np

# Function to compute 2SLS estimator
def iv_2sls(X, Z, Y, include_constant=False):

    if include_constant:
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        Z = np.concatenate((np.ones((Z.shape[0],1)),Z),axis=1)

    # Make transposes
    XT = np.transpose(X)
    ZT = np.transpose(Z)

    # Make estimator
    first_part = np.linalg.inv(XT @ Z @ np.linalg.inv(ZT @ Z) @ ZT @ X)
    second_part = XT @ Z @ np.linalg.inv(ZT @ Z) @ ZT @ Y
    estimator = np.matmul(first_part, second_part)

    return estimator
