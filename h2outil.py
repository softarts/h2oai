#Import H2O and other libaries that will be used in this tutorial
import h2o
from h2o.estimators import *
from h2o.grid import *
import time

def init():
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
    return train, valid, test, x, y
