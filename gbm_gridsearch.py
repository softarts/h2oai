"""
https://training.h2o.ai/products/1a-introduction-to-machine-learning-with-h2o-3-classification
task 6
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
# GBM GRIDSEARCH
# ===================================

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
print(tuned_gbm_per.auc())
print(tuned_gbm_per.F1())

#gbm_grid.predict(valid)
#gbm_grid_perf=gbm_grid.model_performance(valid)

