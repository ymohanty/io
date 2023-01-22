import numpy as np

class blpmodel:
    
    # Set up an instance of this class
    def __init__(self, data, modeltype, estimatortype):
        # Get data
        self.data = data
        
        # Get modeltype
        self.modeltype = modeltype
        
        # Get estimatortype
        self.estimatortype = estimatortype
        
        # Get estimopts
        self.setEstimOpts()
        
    def setEstimOpts(self):
        # Price sd is estimated normally in first stage
        self.estimopts.pricesd = "normal"
        self.estimopts.price_distr = "normal"
        
        # Monte carlo integration
        self.estimopts.integral_method = "monte_carlo"   

        # Set the number of simulations
        self.estimopts.S = 100
        
        # Tolerance for contraction
        self.estimopts.delta_tol = 1e-12
        self.estimopts.delta_max_iter = 1000
        
        # Tolerance for numerical Jacobian calculation
        self.estimopts.jac_tol = 1e-8
        
        # Initial vectors of mean indirect utilities 
        self.estimopts.delta_init = np.zeros((self.data.dims.J, self.data.dims.T))
        self.estimopts.delta.init(np.isnan(self.data.s_jt)) = np.nan
        
        # Seed for simulation
        self.estimopts.stream = np.random.seed("mt19937ar")
        
        # Initialize the beta 
        self.estimopts.beta_init()

    # Method to draw random shocks - not sure this is correct
    def drawrandomshocks(self):
        self.nu = np.reshape(np.random.standard_normal(self.num_rc * self.estimopts.S), (self.num_rc, self.estimopts.S))