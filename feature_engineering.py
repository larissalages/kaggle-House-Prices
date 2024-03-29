import collections
import pandas as pd
from sklearn.feature_selection import VarianceThreshold 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# drop if exists
def drop_cols(df, del_cols):   
    for col in (set(del_cols) & set(df.columns)):
        df = df.drop([col], axis=1)
    return df

def outlier_detection(method, X):
	if method == 'DBSCAN':
		outlier_detection = DBSCAN(min_samples = 3, eps = 0.5)
		clusters = outlier_detection.fit_predict(X)
		return clusters


def feature_selection(X, X_kaggle, method, porcentage=1, n_comp=1, k =1, y = []):
	new_X = X
	new_X_kaggle = X_kaggle	
	if method == 'VarianceThreshold':
		var = VarianceThreshold(threshold=(porcentage * (1 - porcentage)))
		new_X = var.fit_transform(X)
		new_X_kaggle = var.transform(X_kaggle)
		return new_X, new_X_kaggle, ''

	if method == 'PCA':
		pca = PCA(n_components=n_comp, svd_solver = 'full')
		new_X = pca.fit_transform(X)
		new_X_kaggle = pca.transform(X_kaggle)
		return new_X, new_X_kaggle, pca.explained_variance_ratio_

	if method == 'SelectKBest':
		kbest = SelectKBest(k)
		new_X = kbest.fit_transform(X, y)
		new_X_kaggle = kbest.transform(X_kaggle)
		return new_X, new_X_kaggle, ''		

# Remove outliers training data
def remove_outliers(data):
	# Remove outliers
	data.drop(data[(data['OverallQual']<5) & (data['SalePrice']>200000)].index, inplace=True)
	data.drop(data[(data['GrLivArea']>4500) & (data['SalePrice']<300000)].index, inplace=True)
	data.reset_index(drop=True, inplace=True)

	return data

def remove_useless_features(data):
	data = data.drop(['Utilities', 'MiscFeature'], axis=1)
	return data

#Replace values where NaN has meaning
def replace_NaN_meaning(data):
# columns where NaN values have meaning e.g. no pool etc.
	cols_fillna = ['PoolQC','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
    # replace 'NaN' with 'None' in these columns
	for col in cols_fillna:
		data[col].fillna('None',inplace=True)
	
	# GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
	for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
		data[col] = data[col].fillna('None')

	#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
		data[col] = data[col].fillna(0)
    
	#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		data[col] = data[col].fillna(0)

	# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
	for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		data[col] = data[col].fillna('None')
    
	#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	data["MasVnrType"] = data["MasVnrType"].fillna("None")
	data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

	#Functional : data description says NA means typical
	data["Functional"] = data["Functional"].fillna("Typ")

	# MSSubClass : Na most likely means No building class. We can replace missing values with None
	data['MSSubClass'] = data['MSSubClass'].fillna("None")

	data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
	return data

def transform_numerical_col_categorical(data):
	#MSSubClass=The building class
	data['MSSubClass'] = data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	#data['OverallCond'] = data['OverallCond'].astype(str)

	#Year and month sold are transformed into categorical features.
	#data['YrSold'] = data['YrSold'].astype(str)
	#data['MoSold'] = data['MoSold'].astype(str)

	return data

def find_numerical_categorical_columns(X):
    num_columns = X.select_dtypes(exclude=['object'])
    categ_columns = X.select_dtypes(['object'])
    return num_columns, categ_columns

def replace_NaN_numerical(X, num_columns):
    #Replace NAN in numerical column data by the mean of the column
    X[num_columns.columns] = X[num_columns.columns].groupby(num_columns.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
    return X[num_columns.columns]

def most_frequent_word(col):
    col = [x for x in col if str(x) != 'nan']
    counter = collections.Counter(col)
    return counter.most_common()[0][0]

def replace_NaN_categ(X, categ_columns):
    for col in categ_columns:
        X[col].fillna(most_frequent_word(col),inplace=True)
    return X[categ_columns]

def handle_missing_data(X):
	#Find Numerical and Categorical Columns
	num_columns, categ_columns = find_numerical_categorical_columns(X)

	#Handle NaN values in numerical data
	X[num_columns.columns] = replace_NaN_numerical(X, num_columns)

	#Handle NaN values in categorical data
	X[categ_columns.columns] = replace_NaN_categ(X, categ_columns.columns)

	return X, num_columns, categ_columns

def adding_features(data):
	# Since area related features are very important to determine house prices, add one more feature which is the total area of basement, first and second floor areas of each house
	# Adding total sqfootage feature 
	data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
	data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
	data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
	data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
	data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
	data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
	data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
	return data

def label_enconding(X):
	
	cols_label = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
			'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
			'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
			'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
			'YrSold', 'MoSold']
	# process columns, apply LabelEncoder to categorical features
	for c in cols_label:
		lbl = LabelEncoder() 
		lbl.fit(list(X[c].values)) 
		X[c] = lbl.transform(list(X[c].values))
	
	return X, cols_label


def split_data_dataKaggle(X, X_kaggle, one_hot_encoding_all, num_columns, labelEncod_col):
	# Split again between the training kaggle data and test kaggle data
	one_hot_encoding = one_hot_encoding_all[:X.shape[0]]
	one_hot_encoding_kaggle = one_hot_encoding_all[X.shape[0]:]

	#Join categorical and numerical columns again
	X_final = pd.concat([ X[num_columns.columns], X[labelEncod_col], one_hot_encoding], axis=1)
	X_final_kaggle = pd.concat([ X_kaggle[num_columns.columns], X_kaggle[labelEncod_col], one_hot_encoding_kaggle], axis=1)

	return X_final.dropna(), X_final_kaggle.dropna()

