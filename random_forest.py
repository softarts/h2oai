from h2outil import *

def rf(train, valid, test, x, y):
    print("# ===================================")
    print("# RANDOM FOREST")
    print("# ===================================")

    rf = H2ORandomForestEstimator (seed = 42, model_id = 'default_rf')
    rf.train(x = x, y = y, training_frame = train, validation_frame = valid)
    print(rf)

    print("accuracy %s" % rf.accuracy())
    print(rf.predict(valid).head(10))
    
    rf_perf=rf.model_performance(valid)
    print("AUC %s" % rf_perf.auc())
    print("F1 %s" % rf_perf.F1())    
    return rf_perf

if __name__ == "__main__":
    train, valid, test, x, y = init()
    rf(train, valid, test, x, y)
    pass