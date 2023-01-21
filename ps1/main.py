import data.blpdata as blpdata
import data
import sys


def main(args):

    # Load BLP data into data class
    dataobj = blpdata.Data(data.BLP_DATA_LOC)


if __name__ == '__main__':
    main(sys.argv)
