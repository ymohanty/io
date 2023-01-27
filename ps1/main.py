import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import numpy as np
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.BLP_DATA_LOC,spec="blp",add_outside_good=False)
    print(dataobj.micro_data)
    modelobj = blpmodel.Model(dataobj,"gmm")
    print(modelobj.data.dims)
    print(np.amax(modelobj.data.s))
    print(np.amin(modelobj.data.s))

    modelobj.get_delta(modelobj.beta_o_hat, modelobj.beta_u_hat)


if __name__ == '__main__':
    main(sys.argv)
