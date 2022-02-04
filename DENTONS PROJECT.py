import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

import seaborn as sns

sns.set(style="white", color_codes=True)

df = pd.read_csv(r'C:/Users/kwake/Desktop/loandoc.csv')
df.head()

df.describe()
df.info()


# DATA Preprocessing
# finding the missing values
df.isnull().sum()

# filling the missing numerical variables with the mean
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

# filling the missing categorial variables with the mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df.isnull().sum()

# Exploring each variable - Categorical Attributes
sns.countplot(df['Gender'])
sns.countplot(df['Married'])
sns.countplot(df['Dependents'])
sns.countplot(df['Education'])
sns.countplot(df['Self_Employed'])
sns.countplot(df['Property_Area'])
sns.countplot(df['Loan_Status'])

sns.pairplot(df[['Gender']])


# Exploring each variable - Numerical Attributes
sns.displot(df["ApplicantIncome"])
df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
sns.displot(df["ApplicantIncomeLog"])

sns.displot(df["CoapplicantIncome"])
df['CoapplicantIncomeLog'] = np.log(df['CoapplicantIncome'])
sns.displot(df["CoapplicantIncomeLog"])

sns.displot(df["LoanAmount"])
df['LoanAmount'] = np.log(df['LoanAmount'])

sns.displot(df["Loan_Amount_Term"])
df['Loan_Amount_TermLog'] = np.log(df['Loan_Amount_Term'])
sns.displot(df["Loan_Amount_TermLog"])

sns.displot(df["Credit_History"])

# CREATION OF NEW ATTRIBUTES
# total income
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
sns.displot(df["Total_Income"])
df['Total_IncomeLog'] = np.log(df['Total_Income'])
sns.displot(df["Total_IncomeLog"])

df.head()



# COORELATION MATRIX
corr = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(corr, annot=True, cmap="BuPu")

# Removing/ dropping unnecessary columns/ variables
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Loan_ID',
        'CoapplicantIncomeLog']
df = df.drop(columns=cols, axis=1)
df.head()

# Converting categorial variable into numeric to work the model
from sklearn.preprocessing import LabelEncoder

cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.head()

# filling missing values in Total_IncomeLog
df['Total_IncomeLog'] = df['Total_IncomeLog'].fillna(df['Total_IncomeLog'].mean())
df.head()
# df.isnull().sum() confirmed no missing values
df['Credit_History'] = df['Credit_History'].astype(int)
df['ApplicantIncomeLog'] = df['ApplicantIncomeLog'].astype(int)
df['Loan_Amount_TermLog'] = df['Loan_Amount_TermLog'].astype(int)
df['Total_IncomeLog'] = df['Total_IncomeLog'].astype(int)

# df.info()


# Train-Test Split
# specifying input x and output variables y
x = df.drop(columns=['Loan_Status'], axis=1)
y = df['Loan_Status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model Training
# classify function
from sklearn.model_selection import cross_val_score

def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test) * 100)
    # cross validation is used for better validation of the model
    score = cross_val_score(model, x, y, cv=5)
    print("Cross Validation is,", np.mean(score) * 100)

# now running the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
classify(model, x, y)

# using Decision Tree model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
classify(model, x, y)

# using RandomForest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
classify(model, x, y)

# Using Confusion Matrix
model = RandomForestClassifier()
model.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix

y_prediction = model.predict(x_test)
cm = confusion_matrix(y_test, y_prediction)
print(cm)

sns.heatmap(cm, annot=True)

