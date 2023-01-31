import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data



def main():

    # Load micro data into data class and estimate
    dataobj = blpdata.Data(data.MICRO_DATA_LOC,spec="micro",add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "mle")
    modelobj.estimate()

    # Load logit data into data class and estimate
    dataobj = blpdata.Data(data.LOGIT_DATA_LOC, spec="logit", add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "2sls")
    modelobj.estimate()
    print(modelobj.compute_elasticities())

    # Load blp data into data class and estimate
    dataobj = blpdata.Data(data.BLP_DATA_LOC, spec="blp", add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "gmm")
    modelobj.estimate()
    print(modelobj.compute_elasticities())




if __name__ == '__main__':
    main()
