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
print(sorted_rf_depth)

"""
Hyper-Parameter Search Summary: ordered by decreasing auc
    max_depth    model_ids            auc
--  -----------  -------------------  --------
    10           depth_grid_model_8   0.848305
    12           depth_grid_model_9   0.847563
    9            depth_grid_model_7   0.847256
    13           depth_grid_model_10  0.846976
    8            depth_grid_model_6   0.845329
    15           depth_grid_model_11  0.843052
    7            depth_grid_model_5   0.842525
    6            depth_grid_model_4   0.836842
    5            depth_grid_model_3   0.831181
    20           depth_grid_model_12  0.826483
    25           depth_grid_model_13  0.81836
    3            depth_grid_model_2   0.815502
    35           depth_grid_model_14  0.813746
    1            depth_grid_model_1   0.76863
"""