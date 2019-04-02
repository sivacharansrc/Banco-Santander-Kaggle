import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', False)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# IMPORTING THE TRAINING FILE
df_train = pd.read_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\input\\train.csv")
df_test = pd.read_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\input\\test.csv")

df_train.head()
# sns.countplot(x='target', data=df_train, palette = 'dark') # The data is heavily imbalanced

df_train.drop('ID_code', axis=1, inplace=True)
y = df_train['target']
x = df_train.drop('target', axis=1)

# IN ORDER TO PERFORM GUASSIAN MODEL, WE NEED TO MAKE SURE THEY ARE DISTRIBUTED NORMALLY AND THE PREDICTORS ARE INDEPENDENT OF EACH OTHER

x[y==1].plot.kde(ind=100, legend=False)
x[y==0].plot.kde(ind=100, legend=False)

# The kernel density estimation shows a different distribution for the binomial classifiers. Let us see if standardizing the features fixes this issue

# STANDARDIZING THE DATA SET
from sklearn.preprocessing import PowerTransformer

standardizeData = PowerTransformer(method='yeo-johnson', standardize=True)

x_cols = x.columns
x = standardizeData.fit_transform(x)
x = pd.DataFrame(data=x, columns=x_cols)

plt.subplot(211)
x[y==1].plot.kde(ind=100, legend=False)
plt.title('Classifier=1')
plt.subplot(212)
x[y==0].plot.kde(ind=100, legend=False)
plt.title('Classifier=0')
plt.show()


# STANDARDIZING FEATURES USING QUANTILE TRANSFORMER
from sklearn.preprocessing import QuantileTransformer

quantileTransformer = QuantileTransformer(output_distribution='normal')
colNames = x_train.columns

x_train = quantileTransformer.fit_transform(x_train)
x_train = pd.DataFrame(data=x_train, columns=colNames)

x_validation = quantileTransformer.fit_transform(x_validation)
x_validation = pd.DataFrame(data=x_validation, columns=colNames)


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

# APPLYING THE MODEL TO TEST DATA
ID = df_test.ID_code
df_test.drop('ID_code', axis=1, inplace=True)

df_test = standardizeData.fit_transform(df_test)
df_test = pd.DataFrame(data=df_test, columns=x_cols)


predictions = model.predict(df_test)
output = pd.DataFrame({'ID_code': ID, 'target': predictions})
output.to_csv("C:\\Users\\sivac\\Documents\\Python Projects\\Banco Santander Kaggle\\output\\Submission 8 - Naive Bayes Using yeo-johnson Transformation.csv", index=False )
