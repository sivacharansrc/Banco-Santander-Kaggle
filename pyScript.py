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
sampleData = x_train
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