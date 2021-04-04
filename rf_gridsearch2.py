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
# RF GRIDSEARCH
# ===================================

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
print(sorted_rf_depth)

"""
Hyper-Parameter Search Summary: ordered by decreasing auc
    max_depth    sample_rate    model_ids               auc
--  -----------  -------------  ----------------------  --------
    12           0.51           rf_random_grid_model_1  0.853431
    11           0.95           rf_random_grid_model_6  0.85234
    11           0.77           rf_random_grid_model_9  0.851552
    10           0.71           rf_random_grid_model_5  0.851441
    11           0.35           rf_random_grid_model_3  0.851292
    9            0.29           rf_random_grid_model_7  0.849524
    9            0.38           rf_random_grid_model_4  0.849234
    9            0.84           rf_random_grid_model_8  0.848888
    8            0.42           rf_random_grid_model_2  0.846969
"""

tuned_rf = sorted_rf.models[0]
tuned_rf_per = tuned_rf.model_performance(valid)
tuned_rf_per.auc()
tuned_rf_per.F1()

# print("Default RF AUC: %.4f \nTuned RF AUC:%.4f" % (default_rf_per.auc(), tuned_rf_per.auc()))
