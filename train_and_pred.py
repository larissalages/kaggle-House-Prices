import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


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
		grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=10, verbose=1)
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_GradientBoostingRegressor(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = GradientBoostingRegressor(max_features='sqrt',loss='huber').fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'loss' : ['ls', 'lad', 'huber', 'quantile'], 'learning_rate' : [0.01, 0.1, 1],'n_estimators' : [100,500, 1000]}
		grid_rf = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=10)
		reg = grid_rf.fit(X_train, y_train)
		y_pred = reg.predict(X_test)

	return reg, y_pred

def train_pred_MLPRegressor(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = MLPRegressor().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'hidden_layer_sizes' : [100,(100,50),(100,50,20)], 'solver' : ['lbfgs', 'sgd', 'adam'],  }
		grid_rf = GridSearchCV(MLPRegressor(learning_rate = 'adaptive', activation = 'logistic'), param_grid, cv=10)
		reg = grid_rf.fit(X_train, y_train)

	return reg, y_pred

def train_pred_SVR(X_train, y_train, X_test, grid_search=False):
	if grid_search==False:
		reg = SVR().fit(X_train, y_train)
		y_pred = reg.predict(X_test)
	else:
		param_grid = {'kernel' : ['rbf','sigmoid'], 'C' : [0.01,0.1,1,10,100,1000], 'gamma': [0.01,0.1,1,10,100]  }
		grid_rf = GridSearchCV(SVR(), param_grid, cv=10)
		reg = grid_rf.fit(X_train, y_train)

	return reg, y_pred