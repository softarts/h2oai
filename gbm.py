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
# GBM
# ===================================

model = H2OGradientBoostingEstimator(seed = 42, model_id = 'default_gbm')
model.train(x = x, y = y, training_frame = train, validation_frame = valid)
print(model)


model.plot(metric='auc')
model.varimp_plot(20)
model.accuracy()
model.F1()

print("== PREDICT ===================================")
model.predict(valid).head(20)
default_model_perf=model.model_performance(valid)
print(default_model_perf.auc())
