import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def score(y_test,y_pred):
    #### Score using RMSE (root mean square error)
    # Using log to make sure errors in predicting expensive houses and cheap houses will affect the result equally
    rmse = np.sqrt(mean_squared_error(np.log(y_test), np.log(y_pred)))
    #mse = mean_squared_error(y_test, y_pred)
    #rmse = sqrt(mse)
    print("RMSE score: %f" % rmse)
    
    #### Score using MAPE (mean absolute porcentage error)
    ###### (MAPE is how far the models predictions are off from their corresponding outputs on average)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("MAPE score: %f" % mape)