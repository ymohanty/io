import numpy as np
import pandas as pd


def generate_data(T, filename="ps2q2.csv"):
    """
    Simulate entry data in T markets

    :param T: Number of markets
    """


    # Define function to recover equilibrium entrants
    def n_star(x_t,phi_it,theta):

        # Sort the idiosyncratic profitability within markets
        # from highest to lowest
        phi_it = np.reshape(phi_it, (T,K))
        x_t = np.reshape(x_t, (T,K))
        keys = np.argsort(-phi_it)
        phi_it = np.take_along_axis(phi_it, keys, axis=1)
        n = np.reshape(range(1,K+1),(1,K))

        # Find profitable firms
        pi_it = v(x_t, n, theta) + phi_it
        profitable = pi_it > 0
        n_star = np.sum(profitable, axis=1)

        # Reshape variables
        profitable = np.reshape(np.take_along_axis(profitable.astype(int), np.argsort(keys,axis=1), axis=1), (T*K,1)).flatten()
        n_star = np.reshape(np.repeat(n_star, K), (T*K,1)).flatten()
        pi_it = np.reshape(np.take_along_axis(pi_it, np.argsort(keys,axis=1),axis=1),(T*K,1)).flatten()

        return n_star, profitable, pi_it




    # Define function to recover variable profits as a function of number of entrants
    # and market characteristics.
    def v(x_t,n,theta):
        return theta['gamma'] + x_t*theta['beta'] - np.log(n)*theta['delta']

    # Set seed
    rng = np.random.default_rng(2023)

    # Number of potential entrants
    K = 30

    # Define parameters
    alpha = 1
    beta = 2
    delta = 6
    gamma = 3
    rho = 0.8
    theta = {
        'alpha':alpha,
        'beta':beta,
        'delta':delta,
        'gamma':gamma,
        'rho':rho
    }

    # Market characteristics
    x_t = rng.normal(0,1,T)
    x_t = np.repeat(x_t,K)

    # Firm characteristics
    z_it = rng.normal(0,2,K)
    z_it = np.tile(z_it,T)

    # Shocks
    eta_t = rng.normal(0,1,T)
    eta_t = np.repeat(eta_t,K)

    eps_it = rng.normal(0,1,K)
    eps_it = np.tile(eps_it,T)

    # Profits and entry
    phi_it = z_it*alpha + rho*eta_t + np.sqrt((1-rho)**2)*eps_it
    n_star, profitable, pi_it = n_star(x_t, phi_it, theta)

    # Construct pandas dataframe
    t = range(1,T+1)
    t = np.repeat(t,K)
    i = range(1,K+1)
    i = np.tile(i,T)
    df = {'t':t,'i':i, 'x_t':x_t, 'z_it':z_it, 'num_firms':n_star, 'entered':profitable}
    df = pd.DataFrame(df)
    df.to_csv(filename,index=False)





if __name__ == '__main__':
    generate_data(100)

