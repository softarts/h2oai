from h2outil import *

def glm_gridsearch(train, valid, test, x, y):
    print("# ===================================")
    print("# GLM GRIDSEARCH2")
    print("# ===================================")

    glm_grid = h2o.grid.H2OGridSearch (   
        H2OGeneralizedLinearEstimator(family = "binomial",
                                    lambda_search = True),    
    hyper_params = {"alpha": [x*0.01 for x in range(0, 100)],
                    "missing_values_handling" : ["Skip", "MeanImputation"]},    
    grid_id = "glm_random_grid",    
    search_criteria = {
        "strategy":"RandomDiscrete",
        "max_models":300,
        "max_runtime_secs":300,
        "seed":42})


    glm_grid.train(x = x, y = y, training_frame = train, validation_frame = valid)
    sorted_glm_grid = glm_grid.get_grid(sort_by = 'auc', decreasing = True)
    sorted_glm_grid.sorted_metric_table()
    print("==== SORTED RF TABLE ======================")
    print(sorted_glm_grid)

    print("==== PREDICT ======================")
    tuned_glm = sorted_glm_grid.models[0]
    print(tuned_glm.summary())
    tuned_per = tuned_glm.model_performance(valid)

    print("AUC %s" % tuned_per.auc())
    print("F1 %s" % tuned_per.F1())    
    return tuned_per

"""
glm.plot(metric='negative_log_likelihood')
glm.varimp_plot()
glm.accuracy()
# glm.accuracy(thresholds = 0.9638505373028652)
glm.predict(valid).head(10)
default_glm_perf=glm.model_performance(valid)
print(default_glm_perf.auc())
"""

if __name__ == "__main__":
    train, valid, test, x, y = init()
    glm_gridsearch(train, valid, test, x, y)
    pass