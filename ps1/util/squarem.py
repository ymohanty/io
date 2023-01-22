# Code to implement the SQUAREM algorithm to speed up the fixed point iteration
# Adapted from matlab function
def squarem(x, f, con_tol=1e-13, max_iter=10000, step_min=1, step_max=1, mstep=4):
    # x is initial guess of the parameter
    # f is the function for fixed point iteration

    # Optional
    # con_tol is the tolerance for convergence (default 1e-13)
    # max_iter: maximum number of iterations before failure (default 10000)
    # step_min: initial step min (default 1)
    # step_max: initial step max (default 1)
    # mstep: accepted step scaling factor (default 4)