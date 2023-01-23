import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.BLP_DATA_LOC,spec="blp")
    print(dataobj.dims["K_1"])
    modelobj = blpmodel.Model(dataobj,"gmm")
    print(modelobj.beta)


if __name__ == '__main__':
    main(sys.argv)
