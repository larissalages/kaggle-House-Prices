import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def train_pred_LinearRegression(X_train, y_train, X_test, grid_search=False):
	reg = LinearRegression().fit(X_train, y_train)
	y_pred = reg1.predict(X_test)

	return reg, y_pred

def train_pred_DecisionTree(X_train, y_train, X_test, grid_search=False):
	reg = DecisionTreeRegressor().fit(X_train, y_train)
	y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_RandomFlorest(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = RandomForestRegressor().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100] }
		grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_GradientBoostingRegressor(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = GradientBoostingRegressor(max_features='sqrt',loss='huber').fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'loss' : ['ls', 'lad', 'huber', 'quantile'], 'learning_rate' : [0.01, 0.1, 1],'n_estimators' : [100,500, 1000]}
		grid_rf = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_MLPRegressor(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = MLPRegressor().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'hidden_layer_sizes' : [100,(100,50),(100,50,20)], 'solver' : ['lbfgs', 'sgd', 'adam'],  }
		grid_rf = GridSearchCV(MLPRegressor(learning_rate = 'adaptive', activation = 'logistic'), param_grid, cv=10, n_jobs=-1)
		reg = grid_rf.fit(X_train, y_train)

	return reg, y_pred

def train_pred_SVR(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = SVR().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'kernel' : ['rbf','sigmoid'], 'C' : [0.01,0.1,1,10,100,1000], 'gamma': [0.01,0.1,1,10,100]  }
		grid_rf = GridSearchCV(SVR(), param_grid, cv=10, n_jobs=-1)
		reg = grid_rf.fit(X_train, y_train)

	return reg, y_pred

# Recursive feature elimination with cross-validation	
def train_pred_RFECV(X_train, y_train, X_test, grid_search=False):
	# Create the RFE object and compute a cross-validated score.
	rf = RandomForestRegressor()
	# The "accuracy" scoring is proportional to the number of correct
	# classifications
	rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(2))
	reg = rfecv.fit(X_train, y_train)
	print("Optimal number of features : %d" % rfecv.n_features_)

	return reg

def train_pred_Lasso(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = linear_model.Lasso().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'alpha' : [0.0001,0.001,0.01,0.1,1,10]  }
		grid_rf = GridSearchCV(linear_model.Lasso(), param_grid, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_XGboost(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = xgb.XGBRegressor().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'max_depth': [2,4,6,7,10], 'n_estimators': [50,100,200,500]}
		grid_rf = GridSearchCV(xgb.XGBRegressor(early_stopping_rounds=5), param_grid, cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred	

def train_pred_LightGBM(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = lgb.LGBMRegressor(objective='regression',num_leaves=5,
								learning_rate=0.05, n_estimators=720,
								max_bin = 55, bagging_fraction = 0.8,
								bagging_freq = 5, feature_fraction = 0.2319,
								feature_fraction_seed=9, bagging_seed=9,
								min_data_in_leaf =6, min_sum_hessian_in_leaf = 11).fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'max_depth': [2,4,6,7,10], 'min_data_in_leaf': [6, 10, 20], 'feature_fraction': [0.2319, 0.5, 0.7, 0.8], 'bagging_fraction': [0.8, 0.9, 0.95, 0.99]}
		grid_rf = GridSearchCV(lgb.LGBMRegressor(objective='regression'), param_grid, cv=10, n_jobs=-1, scoring="neg_mean_squared_error")
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred


def train_comb_predictor(X_train, y_train, list_alg, list_predictors):
    dict_ = {}
    for i in range(len(list_alg)):
        dict_[list_alg[i]] = list_predictors[i].predict(X_train)

    X_comb = pd.DataFrame(dict_)
    reg_comb = LinearRegression().fit(X_comb, y_train)

    return reg_comb

def test_comb_predictor(reg_comb, list_alg, list_predictions):
    dict_ = {}
    for i in range(len(list_alg)):
        dict_[list_alg[i]] = list_predictions[i]

    X_comb_test = pd.DataFrame(dict_)
    y_pred_ens = reg_comb.predict(X_comb_test)

    return y_pred_ens

