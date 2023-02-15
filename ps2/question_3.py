import pandas as pd
import numpy as np
import data

# Function to make K x K transition matrix
# Heavily influenced by https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
def transition_matrix(x, k, d, replacement = True):
    if replacement:
        # Discretize the array x into K chunks
        bins = np.linspace(0, np.amax(x), num=k)
        discrete = np.digitize(x, bins)

        # Remove all extraneous information
        for i in range(np.size(d, 0)):
            if d[i, 0] == 0 and d[i-1, 0]==0:
                discrete[i, 0] = k+1
        discrete[0, 0] = k+1
        discrete_flat = discrete.flatten() - 1

        # Create matrix of zeros of size K + 1
        matrix_long = np.zeros((k+1, k+1))

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
                matrix_return[i] = matrix[i]/matrix[i].sum(keepdims=True)
        return matrix_return
    else:
        # Discretize the array x into K chunks
        bins = np.linspace(0, np.amax(x), num=k)
        discrete = np.digitize(x, bins)

        # Remove all extraneous information
        for i in range(np.size(d, 0)):
            if d[i, 0] == 0 and d[i-1, 0]==0:
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
            utility[i, 0] = -theta[0]*x[i, 0] - theta[1]*(x[i, 0]/100)**2
        if d[i, 0] == 1:
            utility[i, 0] = -theta[2]
    return utility

def main():
    # Read in data and convert to arrays
    raw_data = pd.read_csv(data.DATA_LOC_3)
    data_array = raw_data.to_numpy()

    # Make decision array and fill it in
    decision_array = np.zeros(data_array.shape)
    for i in range(np.size(data_array, 0) - 1):
        if data_array[i+1, 0] < data_array[i, 0]:
            decision_array[i, 0] = 1

    # Make transition matrices
    trans_repair = transition_matrix(data_array, 20, decision_array, replacement=False)
    #print(trans_repair)
    trans_replaced = transition_matrix(data_array, 20, decision_array, replacement=True)
    #print(trans_replaced)

    # Discretize x
    bins = np.linspace(0, np.amax(data_array), num=20)
    discrete_x = np.digitize(data_array, bins)

if __name__ == '__main__':
    main()