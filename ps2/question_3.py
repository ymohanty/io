import pandas as pd
import numpy as np
import data


# Function to make K x K transition matrix
# Heavily influenced by https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
def transition_matrix(x, k, d, replacement=True):
    """
    Estimate the transition matrix of Harold Zurcher's
    dynamic problem.

    :param x: Data on mileage in Zurcher's problem (T x 1)
    :param k: Number of bins to discretize state space
    :param d: Data on engine replacement. d in {0,1} (T x 1)
    :param replacement: Return the transition matrix for d = 1.
    :return: The transition matrix (K x K)
    """
    if replacement:
        # Discretize the array x into K chunks
        bins = np.linspace(0, np.amax(x), num=k)
        discrete = np.digitize(x, bins)

        # Remove all extraneous information
        for i in range(np.size(d, 0)):
            if d[i, 0] == 0 and d[i - 1, 0] == 0:
                discrete[i, 0] = k + 1
        discrete[0, 0] = k + 1
        discrete_flat = discrete.flatten() - 1

        # Create matrix of zeros of size K + 1
        matrix_long = np.zeros((k + 1, k + 1))

        # Count transitions between states
        for (i, j) in zip(discrete_flat, discrete_flat[1:]):
            matrix_long[i][j] += 1

        # Change it to size K
        intermed = np.delete(matrix_long, k, 0)
        matrix = np.delete(intermed, k, 1)

        # Make everything a probability
        matrix_return = np.zeros((k, k))
        for i in range(k):
            if matrix[i].sum() == 0:
                matrix_return[i] = matrix[i]
            else:
                matrix_return[i] = matrix[i] / matrix[i].sum(keepdims=True)
        return matrix_return
    else:
        # Discretize the array x into K chunks
        bins = np.linspace(0, np.amax(x), num=k)
        discrete = np.digitize(x, bins)

        # Remove all extraneous information
        for i in range(np.size(d, 0)):
            if d[i, 0] == 0 and d[i - 1, 0] == 0:
                discrete[i, 0] = k + 1
        discrete[0, 0] = k + 1
        discrete_flat = discrete.flatten() - 1

        # Create matrix of zeros of size K + 1
        matrix_long = np.zeros((k + 1, k + 1))

        # Count transitions between states
        for (i, j) in zip(discrete_flat, discrete_flat[1:]):
            matrix_long[i][j] += 1

        # Change it to size K
        intermed = np.delete(matrix_long, k, 0)
        matrix_only_replaces = np.delete(intermed, k, 1)

        # Now do everything for all transitions
        discrete_2 = np.digitize(x, bins)
        discrete_flat_2 = discrete_2.flatten() - 1
        matrix_long_2 = np.zeros((k, k))
        for (i, j) in zip(discrete_flat_2, discrete_flat_2[1:]):
            matrix_long_2[i][j] += 1

        # Subtract to get the matrix we need
        matrix = matrix_long_2 - matrix_only_replaces

        # Make everything a probability
        matrix_return = np.zeros((k, k))
        for i in range(k):
            if matrix[i].sum() == 0:
                matrix_return[i] = matrix[i]
            else:
                matrix_return[i] = matrix[i] / matrix[i].sum(keepdims=True)
        return matrix_return


# Function to compute utility
def utility(x, d, theta):
    utility = np.zeros(x.shape)
    for i in range(np.size(x, 0)):
        if d[i, 0] == 0:
            utility[i, 0] = -theta[0] * x[i, 0] - theta[1] * (x[i, 0] / 100) ** 2
        if d[i, 0] == 1:
            utility[i, 0] = -theta[2]
    return utility


# Function to compute right hand side of fixed point
def contraction(EV, beta, x, theta, trans_matrix, noisy=True):
    return np.matmul(trans_matrix, np.log(np.exp(utility(x, np.zeros(x.shape), theta) + beta * EV)
                                          + np.exp(utility(x, np.ones(x.shape), theta) + beta * EV)))


def get_value_function(theta, beta, x, trans_matrix):

    diff = np.inf
    niter = 1
    EV = np.zeros(20,2)
    while diff > 1e-13 and niter < 1000:
        old_EV = EV
        EV[:,0] = contraction(old_EV, beta, x, theta, trans_matrix[:,:,0])
        EV[:,1] = contraction(old_EV, beta, x, theta, trans_matrix[:,:,1])

        diff = np.amax(np.abs(EV - old_EV))

    if noisy:
        print(f"Num. iteratates: {niter}")
        print(f"Diff: {diff}")
        print("Convergence success: %s" %(diff < 1e-13))













def main():
    # Read in data and convert to arrays
    raw_data = pd.read_csv(data.DATA_LOC_3)
    data_array = raw_data.to_numpy()

    # Make decision array and fill it in
    decision_array = np.zeros(data_array.shape)
    for i in range(np.size(data_array, 0) - 1):
        if data_array[i + 1, 0] < data_array[i, 0]:
            decision_array[i, 0] = 1

    # Make transition matrices
    trans_repair = transition_matrix(data_array, 20, decision_array, replacement=False)
    trans_replaced = transition_matrix(data_array, 20, decision_array, replacement=True)

    # Convert transition matrices to latex
    trans_repair_df = pd.DataFrame(data=trans_repair)
    trans_repair_df.to_latex(buf=data._OUT_PATH + '/trans_matrix_repair.tex', caption='Transition Matrix for d=0',
                             label='tab:trans0', index=True, escape=False, float_format="%.1f")
    trans_replaced_df = pd.DataFrame(data=trans_replaced)
    trans_replaced_df.to_latex(buf=data._OUT_PATH + '/trans_matrix_replaced.tex', caption='Transition Matrix for d=1',
                               label='tab:trans1', index=True, escape=False, float_format="%.1f")

    # Discretize x
    bins = np.linspace(0, np.amax(data_array), num=20)
    discrete_x = np.digitize(data_array, bins)

    test = np.arange(1, 21)
    test_2 = np.reshape(test, (20, 1))
    # print(test_2)
    print(ev_rhs(np.zeros((20, 1)), 0.999, test_2, [1, 2, 3], trans_replaced))


if __name__ == '__main__':
    main()
