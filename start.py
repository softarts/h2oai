#Import H2O and other libaries that will be used in this tutorial
import h2o
from h2o.estimators import *
from h2o.grid import *
import time

gbm = H2OGradientBoostingEstimator(seed = 42, model_id = 'default_gbm')
h2o.init()

while True: time.sleep(1)