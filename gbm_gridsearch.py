from h2outil import *

def gbm_gridsearch(train, valid, test, x, y):
    print("# ===================================")
    print("# GBM GRIDSEARCH")
    print("# ===================================")

    hyper_params = {'max_depth' : [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15]}

    gbm = H2OGradientBoostingEstimator(model_id = 'grid_gbm', 
                                   ntrees = 50,
                                   seed = 42)

    gbm_grid = H2OGridSearch(gbm, hyper_params,
                         grid_id = 'depth_gbm_grid',
                         search_criteria = {"strategy":"Cartesian"})

    gbm_grid.train(x = x, y = y, training_frame = train, validation_frame = valid)
    print(gbm_grid)
    sorted_gbm_depth = gbm_grid.get_grid(sort_by = 'auc', decreasing = True)
    sorted_gbm_depth.sorted_metric_table()
    print("== SORTED GBM DEPTH ===================================")
    print(sorted_gbm_depth)

    """
    model.plot(metric='auc')
    model.varimp_plot(20)
    model.accuracy()
    model.F1()
    """
    print("== PREDICT ===================================")
    tuned_gbm = sorted_gbm_depth.models[0]
    tuned_gbm_per = tuned_gbm.model_performance(valid)
    print("AUC %s" % tuned_gbm_per.auc())
    print("F1 %s" % tuned_gbm_per.F1())
    return tuned_gbm_per

#gbm_grid.predict(valid)
#gbm_grid_perf=gbm_grid.model_performance(valid)

if __name__ == "__main__":
    train, valid, test, x, y = init()
    gbm_gridsearch2(train, valid, test, x, y)
    pass
