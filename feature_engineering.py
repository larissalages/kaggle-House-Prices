import collections
import pandas as pd

#Replace values where NaN has meaning
def replace_NaN_meaning(data):
# columns where NaN values have meaning e.g. no pool etc.
	cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']
	# replace 'NaN' with 'None' in these columns
	for col in cols_fillna:
		data[col].fillna('None',inplace=True)
	
	#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
		data[col] = data[col].fillna(0)
    
	#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
	for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
		data[col] = data[col].fillna(0)
    
	#MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
	data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

	#Functional : data description says NA means typical
	data["Functional"] = data["Functional"].fillna("Typ")
    
	return data

def transform_numerical_col_categorical(data):
	#MSSubClass=The building class
	data['MSSubClass'] = data['MSSubClass'].apply(str)

	#Changing OverallCond into a categorical variable
	data['OverallCond'] = data['OverallCond'].astype(str)

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

def split_data_dataKaggle(X, X_kaggle, one_hot_encoding_all, num_columns):
	# Split again between the training kaggle data and test kaggle data
	one_hot_encoding = one_hot_encoding_all[:X.shape[0]]
	one_hot_encoding_kaggle = one_hot_encoding_all[X.shape[0]:]

	#Join categorical and numerical columns again
	X_final = pd.concat([ X[num_columns.columns], one_hot_encoding], axis=1)
	X_final_kaggle = pd.concat([ X_kaggle[num_columns.columns], one_hot_encoding_kaggle], axis=1)

	return X_final.dropna(), X_final_kaggle.dropna()

