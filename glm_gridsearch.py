"""
https://training.h2o.ai/products/1a-introduction-to-machine-learning-with-h2o-3-classification
task 2,3,4
"""
#Import H2O and other libaries that will be used in this tutorial
import h2o
from h2o.estimators import *
from h2o.grid import *
import time

h2o.init()

#Import the dataset 
loan_level = h2o.import_file("loan_level_500k.csv")
loan_level.head()
loan_level.describe()
loan_level["DELINQUENT"].table()
train, valid, test = loan_level.split_frame([0.7, 0.15], seed=42)
print("train:%d valid:%d test:%d" % (train.nrows, valid.nrows, test.nrows))

y = "DELINQUENT"

ignore = ["DELINQUENT", "PREPAID", "PREPAYMENT_PENALTY_MORTGAGE_FLAG", "PRODUCT_TYPE"] 

x = sorted(list(set(train.names) - set(ignore)))
print(x)

# ===================================
# GLM GRIDSEARCH
# ===================================

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


"""
glm.plot(metric='negative_log_likelihood')
glm.varimp_plot()
glm.accuracy()
# glm.accuracy(thresholds = 0.9638505373028652)
glm.predict(valid).head(10)
default_glm_perf=glm.model_performance(valid)
print(default_glm_perf.auc())
"""