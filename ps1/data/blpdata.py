import pandas as pd
import numpy as np


class Data:

    def __init__(self, filename: str, spec="micro", add_outside_good=True):
        """

        :param filename:
        """
        ## Read data
        self.raw_data = pd.read_csv(filename)
        self.agg_data = None
        self.micro_data = None

        ## Declare attributes

        # Metadata
        self.spec = spec                 # Specification used
        self.dims = {}                   # Data dimensions
        self.num_rc = None               # Num. random coefficients
        self.model_vars = {}
        self.add_outside_good = add_outside_good

        # Data matrices
        self.choice = None               # I x 1 (Indiv. choices)
        self.d = None                    # I x D (Indiv. char)
        self.x_1 = None                  # T x J x K_1 (Chars that go into mean utility)
        self.x_2 = None                  # T x J x K_2 (Chars that interact with observed hh. chars)
        self.x_3 = None                  # T x J x K_3 (Chars with unobserved taste variation)
        self.z = None                    # T x J x Z (instruments)
        self.s = None                    # T x J (observed market shares)

        ## Initialize instance

        # Get the specification details
        self.get_specification()

        # Get metadata
        self.get_metadata()

        # Get data matrices
        self.get_data_matrices()

    def get_specification(self):
        """

        :param spec:
        """

        # The BLP routine from ps1_ex4.csv
        if self.spec == "blp":
            self.agg_data = self.raw_data
            self.model_vars = {"i":None,"c": None, "t": "market", "j": "choice", "x_1": ["p", "x"], "x_2": [],
                               "x_3": ["p", "x"],
                               "d": [], "s": "shares", "z": ["z1", "z2", "z3", "z4", "z5", "z6"]
                               }

        # The logit routine for ps1_ex3.csv
        elif self.spec == "logit":
            self.agg_data = self.raw_data
            self.model_vars = {"i":None,"c": None, "t": "market", "j": "Product", "x_1": ["Prices", "x"], "x_2": [],
                               "x_3": [],
                               "d": [], "s": "Shares", "z": ["z"]
                               }
        # The micro data logit routine ps1_ex2.csv
        elif self.spec == "micro":
            self.model_vars = {"i":"i","c": "choice", "t": None, "j": "j", "x_1": ["x.1", "x.2", "x.3"], "x_2":["x.1", "x.2", "x.3"],
                               "x_3": [],
                               "d": ["d.1","d.2"], "s": "s", "z": []
                               }
            self.agg_data, self.micro_data = self.get_working_data()

        # Throw exception here if specification
        # not recognized
        else:
            raise Exception(f"Specification {spec} not recognized")

    def get_metadata(self):
        """

        """
        # Counts for index variables
        try:
            self.dims["T"] = len(self.agg_data[self.model_vars["t"]].unique())
        except KeyError:
            print("Micro data found: setting number of markets to 1...")
            self.dims["T"] = 1
        self.dims["J"] = len(self.agg_data[self.model_vars["j"]].unique())

        # Household and product characteristics
        self.dims["K_1"] = len(self.model_vars["x_1"])
        self.dims["K_2"] = len(self.model_vars["x_2"])
        self.dims["K_3"] = len(self.model_vars["x_3"])
        self.dims["D"] = len(self.model_vars["d"])
        self.dims["Z"] = len(self.model_vars["z"])
        self.num_rc = self.dims["K_3"]

    def get_data_matrices(self):
        """

        """
        # Characteristics with fixed effects
        self.x_1 = self.get_product_char_matrix(self.model_vars["x_1"])

        # Product characteristics that interact with household characteristics
        self.x_2 = self.get_product_char_matrix(self.model_vars["x_2"])

        # Characteristics with heterogeneous random effects
        self.x_3 = self.get_product_char_matrix(self.model_vars["x_3"])

        # Household characteristics
        self.d = self.get_household_char_matrix()

        # Get observed market shares
        self.s = self.get_observed_market_share()

        # Get instruments
        self.z = self.get_instruments()

        if self.add_outside_good:
            self.dims["J"] = self.dims["J"]+1

    def get_product_char_matrix(self, vars):
        """

        :param vars:
        :return:
        """
        data = self.agg_data[vars].to_numpy()
        num_cols = data.shape[1]
        data = np.reshape(data, (self.dims["T"], self.dims["J"], num_cols))

        if self.add_outside_good:
            data_w_outside_good = np.zeros((self.dims["T"] , self.dims["J"]+1, num_cols))
            data_w_outside_good[ :, 1:self.dims["J"] + 1, :] = data
            data = data_w_outside_good

        return data

    def get_household_char_matrix(self):
        """

        """
        if self.spec == "micro":
            return self.micro_data[self.model_vars['d']]
        else:
            return np.zeros((1,0))


    def get_instruments(self):
        return np.reshape(self.agg_data[self.model_vars['z']].to_numpy(),
                          (self.dims["T"], self.dims["J"], self.dims['Z']))



    def get_observed_market_share(self):
        """

        :return:
        """
        data = self.agg_data[self.model_vars["s"]].to_numpy()
        s = np.reshape(data, (self.dims["T"], self.dims["J"]))

        if self.add_outside_good:
            s_w_outside_good = np.zeros((self.dims["T"], self.dims["J"]+1))
            s_w_outside_good[:, 1:self.dims["J"] + 1] = s
            s_w_outside_good[:, 0] = 1 - np.sum(s, axis=1)
            s = s_w_outside_good

        return s

    def get_working_data(self):
        """

        """
        if self.spec != "micro":
            raise Exception(f"No microdata for {self.spec} specifications so we cannot aggregate!")

        # Aggregate raw data into market level data to recover shares and characteristics
        agg_data = self.raw_data.groupby("choice").mean()
        agg_data = agg_data[self.model_vars["x_1"]]
        observed_market_shares = self.raw_data.groupby("choice").size()/len(self.raw_data.index)
        agg_data["s"] = pd.Series(observed_market_shares)
        agg_data.reset_index(inplace=True)
        agg_data = agg_data.rename(columns={"choice":"j"})

        # Collect micro data separately
        indiv_indices = pd.Series(data=list(range(len(self.raw_data.index))),
                                  name=self.model_vars["i"])
        indiv_chars = self.raw_data[self.model_vars["d"]]
        indiv_choices = self.raw_data[self.model_vars["c"]]
        micro_data = pd.concat([indiv_indices,indiv_chars,indiv_choices], axis = 1)

        # Return aggregate and microdata
        return agg_data, micro_data

if __name__ == '__main__':
    pass

