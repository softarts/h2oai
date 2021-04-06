from h2outil import *


def rf_gridsearch(train, valid, test, x, y):
    print("# ===================================")
    print("# RF GRIDSEARCH")
    print("# ===================================")

    hyper_parameters = {"max_depth":[8, 9, 10, 11, 12],
                        'sample_rate': [x/100. for x in range(20,101)]}

    rf = H2ORandomForestEstimator(ntrees = 500,
                                seed = 42,
                                stopping_rounds = 5, 
                                stopping_tolerance = 1e-3, 
                                stopping_metric = "auc",
                                model_id = 'rf_grid')

    grid_id = 'rf_random_grid'

    search_criteria = {"strategy":"RandomDiscrete",
                    "max_models":100,
                    "max_runtime_secs":900,
                    "seed":42}

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

    tuned_rf = sorted_rf_depth.models[0]
    tuned_rf_per = tuned_rf.model_performance(valid)
    tuned_rf_per.auc()
    tuned_rf_per.F1()
    return tuned_rf_per

# print("Default RF AUC: %.4f \nTuned RF AUC:%.4f" % (default_rf_per.auc(), tuned_rf_per.auc()))


if __name__ == "__main__":
    train, valid, test, x, y = init()
    rf_gridsearch2(train, valid, test, x, y)
    pass