import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import numpy as np
from util.utilities import get_lower_triangular
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.LOGIT_DATA_LOC,spec="logit",add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "2sls")
    modelobj.estimate()
    test = modelobj.compute_elasticities()
    print(test)
    #test_2 = modelobj.marginal_costs()
    #print(test_2)

if __name__ == '__main__':
    main(sys.argv)
