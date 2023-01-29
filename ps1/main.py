import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import numpy as np
from util.utilities import get_lower_triangular
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.BLP_DATA_LOC,spec="blp",add_outside_good=False)
    modelobj = blpmodel.Model(dataobj,"gmm")
    #I = np.eye(dataobj.dims['Z'])
    modelobj.estimate()
    test = modelobj.compute_elasticities()
    print(test)

if __name__ == '__main__':
    main(sys.argv)
