import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import seaborn as sns

# IMPORTING THE TRAINING FILE
df_train = pd.read_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\input\\train.csv")
df_test = pd.read_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\input\\test.csv")

df_train.head()
sns.countplot(x='target', data=df_train, palette = 'dark') # The data is heavily imbalanced




df_train.drop('ID_code', axis=1, inplace=True)
y = df_train['target']
x = df_train.drop('target', axis=1)


# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(x,y, stratify=y, random_state=1, test_size=0.25)

# UNDER SAMPLING DATA
sampleData = x_train.copy()
sampleData['target'] = y_train

underSample_length = len(sampleData[sampleData.target == 1])

zero_target_indices = sampleData[sampleData.target == 0].index
random_indices = np.random.choice(zero_target_indices, underSample_length, replace=False)

target_data = sampleData[sampleData.target == 1]
zero_target_data = sampleData[sampleData.index.isin(random_indices)]
x_train = pd.concat([target_data, zero_target_data])

y_train = x_train['target']
x_train = x_train.drop('target', axis=1)

# OVERSAMPLING TRAINING DATA
from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)
columns = x_train.columns
x_train,y_train =os.fit_sample(x_train, y_train)
x_train = pd.DataFrame(data=x_train,columns=columns )
y_train= pd.DataFrame(data=y_train,columns=['target'])

# PERFORMING DIMENSIONALITY REDUCTION

# STANDARDIZING FEATURES USING STANDARD SCALER
from sklearn.preprocessing import StandardScaler

stdScaler = StandardScaler()
colNames = x_train.columns

x_train = stdScaler.fit_transform(x_train)
x_train = pd.DataFrame(data=x_train, columns=colNames)

x_validation = stdScaler.fit_transform(x_validation)
x_validation = pd.DataFrame(data=x_validation, columns=colNames)


from sklearn.decomposition import PCA

pca = PCA(n_components=200)
x_train = pca.fit_transform(x_train)
pca_colNames = colNames[0:200]

x_train = pd.DataFrame(data=x_train, columns=pca_colNames)

np.sum(pca.explained_variance_ratio_)

x_validation = pca.fit_transform(x_validation)
x_validation = pd.DataFrame(data=x_validation, columns=pca_colNames)


# RUNNING A BASIC RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, max_features='sqrt', oob_score=True, random_state=1, bootstrap=True, n_estimators=50, n_jobs=-1)
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# CONFUSION MATRIX AND CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#y_validation = y_validation.values

conf_matrix = confusion_matrix(y_validation, predictions)
class_report = classification_report(y_validation, predictions)

print(classification_report(y_train, model.predict(x_train)))
confusion_matrix(y_train, model.predict(x_train))

conf_matrix
print(class_report)

# RUNNING THE MODEL ON TEST DATA SET
ID = df_test.ID_code
df_test.drop('ID_code', axis=1, inplace=True)

predictions = model.predict(df_test)
output = pd.DataFrame({'ID_code': ID, 'target': predictions})
output.to_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\output\\Submission 3 - UnderSampled Logistic Regression lbfgs Solver l2 Penality.csv", index=False )

# PERFORMING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=3000)
model.fit(x_train_new, y_train)

predictions = model.predict(x_validation_new)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_matrix(y_train, model.predict(x_train_new))
print(classification_report(y_train, model.predict(x_train_new)))

confusion_matrix(y_validation, predictions)
print(classification_report(y_validation, predictions))


coefficients = pd.concat([pd.DataFrame(x_train.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
coefficients.columns = ['Variable', 'Coeff']
coefficients['Abs_Coeff'] = np.abs(coefficients['Coeff'])
coefficients.sort_values('Abs_Coeff', inplace=True, ascending=False)
coefficients.reset_index(inplace=True)
variablesToKeep = coefficients['Variable'][0:100]

x_train_new = x_train.loc[:, variablesToKeep]
x_validation_new = x_validation.loc[:, variablesToKeep]

# Running a Naive Bayes Guassian Model:
# Guassian model because the predictors follow a Normal Distribution

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# Training the model using the training data

model.fit(x_train, y_train)
train_predictions = model.predict(x_train)
predictions = model.predict(x_validation)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Confusion Matrix Comparison
print("Confustion Matrix for Training Data")
print(confusion_matrix(y_train, train_predictions))
print("Confustion Matrix for Test Data")
print(confusion_matrix(y_validation, predictions))

# Classification Report Comparison
print("Classification Report for Training Data")
print(classification_report(y_train, train_predictions))
print("Classification Report for Test Data")
print(classification_report(y_validation, predictions))


import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)




def col_multi(dataframe):
	data = dataframe.copy()
	col_names = data.columns
	n_col = data.shape[1]
	i_cols = range(0, n_col)
	print(i_cols)
	for i in i_cols:
		j_cols = range(i, n_col)
		print(j_cols)
		for j in j_cols:
			col_name = 'multi_' + col_names[i] + '_' + col_names[j]
			data.loc[:, col_name] = data[col_names[i]] * data[col_names[j]]
	return(data)

dummy = pd.DataFrame({'col_A': [1,2,3,4,5], 'col_B': [6,7,8,9,10], 'col_C': [11,12,13,14,15]})
col_multi(dummy)
x = col_multi(x)
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84961
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84632
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/85039
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84974
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83570
# https://www.kaggle.com/jesucristo/santander-magic-lgb-0-901
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82515