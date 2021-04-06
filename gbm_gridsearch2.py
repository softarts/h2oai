from h2outil import *


def gbm_gridsearch2(train, valid, test, x, y):
    print("# ===================================")
    print("# GBM GRIDSEARCH2")
    print("# ===================================")
    gbm = H2OGradientBoostingEstimator(ntrees = 500,
                                   learn_rate = 0.05,
                                   seed = 42,
                                   model_id = 'grid_gbm')

    hyper_params_tune = {'max_depth' : [4, 5, 6, 7, 8],
                        'sample_rate': [x/100. for x in range(20,101)],
                        'col_sample_rate' : [x/100. for x in range(20,101)],
                        'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                        'col_sample_rate_change_per_level': [x/100. for x in range(90,111)]}

    search_criteria_tune = {'strategy': "RandomDiscrete",
                            'max_runtime_secs': 900,  
                            'max_models': 100,  ## build no more than 100 models
                            'seed' : 42}

    random_grid = H2OGridSearch(model = gbm, hyper_params = hyper_params_tune,
                                grid_id = 'random_grid',
                                search_criteria = search_criteria_tune)

    random_grid.train(x = x, y = y, training_frame = train, validation_frame = valid)
    sorted_random_search = random_grid.get_grid(sort_by = 'auc',decreasing = True)
    sorted_random_search.sorted_metric_table()
    print("==== SORTED GBM TABLE ======================")
    print(sorted_random_search)

    print("==== PREDICT ======================")
    tuned_gbm = sorted_random_search.models[0]
    tuned_gbm_per = tuned_gbm.model_performance(valid)
    print("AUC %s" % tuned_gbm_per.auc())
    print("F1 %s" % tuned_gbm_per.F1())
    tuned_gbm_per.confusion_matrix()
    return tuned_gbm_per


if __name__ == "__main__":
    train, valid, test, x, y = init()
    gbm_gridsearch2(train, valid, test, x, y)
    pass