import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


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

