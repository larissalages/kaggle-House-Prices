import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def score(y_test,y_pred):
    #### Score using RMSE (root mean square error)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print("RMSE score: %f" % rmse)
    
    #### Score using MAPE (mean absolute porcentage error)
    ###### (MAPE is how far the model’s predictions are off from their corresponding outputs on average)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("MAPE score: %f" % mape)