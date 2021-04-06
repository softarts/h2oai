from h2outil import *

def rf_gridsearch(train, valid, test, x, y):
    print("# ===================================")
    print("# RF GRIDSEARCH2")
    print("# ===================================")

    hyper_parameters = {'max_depth':[1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 20, 25, 35]}

    rf = H2ORandomForestEstimator(seed = 42,
                                stopping_rounds = 5, 
                                stopping_tolerance = 1e-4, 
                                stopping_metric = "auc",
                                model_id = 'rf')

    grid_id = 'depth_grid'

    search_criteria = {'strategy': "Cartesian"}

    #Grid Search
    rf_grid = H2OGridSearch(model = rf, 
                            hyper_params = hyper_parameters, 
                            grid_id = grid_id, 
                            search_criteria = search_criteria)

    rf_grid.train(x = x, y = y, training_frame = train, validation_frame = valid)
    sorted_rf_depth = rf_grid.get_grid(sort_by = 'auc',decreasing = True)
    sorted_rf_depth.sorted_metric_table()
    print("==== SORTED RF TABLE ======================")
    print(sorted_rf_depth)

    print("==== PREDICT ======================")
    tuned_rf = sorted_rf_depth.models[0]
    tuned_rf_per = tuned_rf.model_performance(valid)
    print("AUC %s" % tuned_rf_per.auc())
    print("F1 %s" % tuned_rf_per.F1())    
    return tuned_rf_per



if __name__ == "__main__":
    train, valid, test, x, y = init()
    rf_gridsearch(train, valid, test, x, y)
    pass