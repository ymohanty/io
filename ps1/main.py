import data.blpdata as blpdata
import model.blpmodel as blpmodel
import data
from pathlib import Path


def main():

    # Load micro data into data class and estimate
    print("============== Estimating the micro data model =============\n\n")
    dataobj = blpdata.Data(data.MICRO_DATA_LOC,spec="micro",add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "mle")
    modelobj.estimate()
    modelobj.print_estimates(filename=data._OUT_PATH + '/tab_mle_micro.tex',
                            title='MLE estimates of demand parameters',
                            label='tab:mleMicro')
    print("=============== END ==============\n\n")

    print("============== Estimating the logit model =============\n\n")
    # Load logit data into data class and estimate
    dataobj = blpdata.Data(data.LOGIT_DATA_LOC, spec="logit", add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "2sls")
    modelobj.estimate()
    modelobj.print_estimates(filename=data._OUT_PATH + '/tab_2sls_logit.tex', title='Logit 2SLS estimates of demand parameters',
                            label='tab:logitDemand')
    modelobj.print_elasticities(filename=data._OUT_PATH + '/tab_logit_elasticites.tex',
                                title=" Logit estimates of average cross-price elasticities", label='tab:logitElasticities',
                                format_float='%.2f')
    print("=============== END ==============\n\n")

    print("============== Estimating the BLP model =============\n\n")
    # Load blp data into data class and estimate
    dataobj = blpdata.Data(data.BLP_DATA_LOC, spec="blp", add_outside_good=False)
    modelobj = blpmodel.Model(dataobj, "gmm")
    modelobj.estimate()
    modelobj.print_estimates(filename=data._OUT_PATH + '/tab_gmm_blp.tex',
                            title='BLP GMM estimates of demand parameters',
                            label='tab:blptDemand')
    modelobj.print_elasticities(filename=data._OUT_PATH + '/tab_blp_elasticites.tex',
                                title=" BLP estimates of average cross-price elasticities", label='tab:blpElasticities',
                                format_float='%.3f')
    print("=============== END ==============\n\n")


if __name__ == '__main__':
    main()
