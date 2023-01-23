import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.MICRO_DATA_LOC,spec="micro")
    print(dataobj.micro_data)
    modelobj = blpmodel.Model(dataobj,"gmm")
    print(modelobj.beta_u_hat)


if __name__ == '__main__':
    main(sys.argv)
