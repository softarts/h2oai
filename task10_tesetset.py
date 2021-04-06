from gbm import *
from gbm_gridsearch2 import *
from gbm_gridsearch import *
from rf_gridsearch import *
from rf_gridsearch2 import *

from glm_gridsearch import *
from glm import *

def run():
    train, valid, test, x, y = init()

    """
    gbm0 = gbm(train, valid, test, x, y)
    gbm1 = gbm_gridsearch(train, valid, test, x, y)
    gbm2 = gbm_gridsearch2(train, valid, test, x, y)
    print("Default GBM AUC: %.4f \nTuned GBM AUC:%.4f" % (gbm1.auc(), gbm2.auc()))
    
    
    rf1 = rf_gridsearch(train, valid, test, x, y)
    rf22 = rf_gridsearch2(train, valid, test, x, y)
    print("Default RF AUC: %.4f \nTuned RF AUC:%.4f" % (rf1.auc(), rf2.auc()))
    """

    glm1 = glm(train, valid, test, x, y)
    glm2 = glm_gridsearch(train, valid, test, x, y)
    print("Default RF AUC: %.4f \nTuned RF AUC:%.4f" % (glm1.auc(), glm2.auc()))

if __name__ == "__main__":
    run()