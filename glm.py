from h2outil import *

def glm(train, valid, test, x, y):
    print("# ===================================")
    print("# GLM GRIDSEARCH2")
    print("# ===================================")

    glm = H2OGeneralizedLinearEstimator(family = "binomial", seed = 42, model_id = 'default_glm')

    glm.train(x = x, y = y, training_frame = train, validation_frame = valid)
    
    # plot function 
    # glm.plot(metric='negative_log_likelihood')
    # glm.varimp_plot()
    # glm.accuracy(thresholds = 0.9638505373028652)
    

    print("accuracy %s" % glm.accuracy())
    print(glm.predict(valid).head(10))

    default_glm_perf=glm.model_performance(valid)    
    print("AUC %s" % default_glm_perf.auc())
    print("F1 %s" % default_glm_perf.F1())    
    return default_glm_perf


if __name__ == "__main__":
    train, valid, test, x, y = init()
    glm(train, valid, test, x, y)
    pass