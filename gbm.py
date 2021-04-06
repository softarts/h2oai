from h2outil import *

def gbm(train, valid, test, x, y):
    print("# ===================================")
    print("# GBM")
    print("# ===================================")

    
    gbm = H2OGradientBoostingEstimator(seed = 42, model_id = 'default_gbm')
    gbm.train(x = x, y = y, training_frame = train, validation_frame = valid)
    print("== GBM MODEL ===================================")
    print(gbm)

    """
    model.plot(metric='auc')
    model.varimp_plot(20)
    model.accuracy()
    model.F1()
    """

    print("== PREDICT ===================================")
    print(gbm.predict(valid).head(20))
    gbm_perf=gbm.model_performance(valid)
    print("AUC %s" % gbm_perf.auc())
    print("F1 %s" % gbm_perf.F1())    
    return gbm_perf

if __name__ == "__main__":
    train, valid, test, x, y = init()
    gbm(train, valid, test, x, y)
    pass