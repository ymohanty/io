import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.BLP_DATA_LOC,spec="blp",add_outside_good=True)
    print(dataobj.micro_data)
    modelobj = blpmodel.Model(dataobj,"gmm")
    print(modelobj.data.dims)

    print(modelobj.get_delta(modelobj.beta_o_hat, modelobj.beta_u_hat))


if __name__ == '__main__':
    main(sys.argv)
