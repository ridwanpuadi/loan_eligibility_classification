import pandas as pd
import numpy as np      # For mathematical calculations
import seaborn as sns     # For data visualization
import matplotlib.pyplot as plt     # For plotting graphs
import warnings     # To ignore any warnings
warnings.filterwarnings('ignore')

# Reading data
loan=pd.read_csv('train_loan.csv')

# Copy original data
loan_original=loan.copy()

#######UNDERSTANDING THE DATA#######
# Check columns
loan.columns

# Print data types for each variable
loan.dtypes

# Look shape dataset
loan.shape

# look at the first few lines
loan.head()

# change the data type credit_history
loan['Credit_History'] = loan['Credit_History'].astype(str)

# unique value categorical features
col_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Status']
for col in col_names:
  print(loan[col].value_counts())
  print('******************************')
  
####### EXPLORATORY DATA ANALYSIS #######
# Descriptive statistics
loan.describe()

print ('skewness loan')
display(loan[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].skew())

print ('kurtosis loan')
display(loan[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']].kurtosis())

# Data visualization
## Univariate analysis
loan['Loan_Status'].value_counts()

### Normalize can be set to True to print proportions instead of number
loan['Loan_Status'].value_counts(normalize=True)

### Target variable
sns.set_theme(style='darkgrid')  # setting theme background
loan['Loan_Status'].value_counts().plot.bar(color=['lightblue', 'lightpink'], title = 'Loan Status Distribution')

### Independent variable (nominal)
plt.figure(1)
plt.subplot(231)
loan['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender', color=['lightblue', 'lightpink'])
plt.subplot(232)
loan['Married'].value_counts(normalize=True).plot.bar(title= 'Married', color=['lightblue', 'lightpink'])
plt.subplot(233)
loan['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed', color=['lightblue', 'lightpink'])
plt.subplot(234)
loan['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History', color=['lightblue', 'lightpink'])
plt.subplot(235)
loan['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area', color=['lightblue', 'lightpink'])
plt.tight_layout()
plt.show()

### Independent variable (Ordinal)
plt.figure(1)
plt.subplot(121)
loan['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents', color=['lightsteelblue'])
plt.subplot(122)
loan['Education'].value_counts(normalize=True).plot.bar(title= 'Education', color=['lightsteelblue'])
plt.tight_layout()
plt.show()

### Independent variable (numerical)
plt.figure(1)
plt.subplot(121)
sns.distplot(loan['ApplicantIncome'], color = 'lightcoral');
plt.subplot(122)
loan['ApplicantIncome'].plot.box(figsize=(16,5), color = 'lightcoral')
plt.show()

loan.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle('')

plt.figure(1)
plt.subplot(121)
sns.distplot(loan['CoapplicantIncome'], color='lightcoral');
plt.subplot(122)
loan['CoapplicantIncome'].plot.box(figsize=(16,5), color='lightcoral')
plt.show()

plt.figure(1)
plt.subplot(121)
df=loan.dropna()
sns.distplot(loan['LoanAmount'], color='lightcoral');
plt.subplot(122)
loan['LoanAmount'].plot.box(figsize=(16,5), color='lightcoral')
plt.show()

## Bivariate analysis
### Categorical independent v target variable
Gender=pd.crosstab(loan['Gender'],loan['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4,), color=['lightblue', 'lightpink'])

Married=pd.crosstab(loan['Married'],loan['Loan_Status'])
Dependents=pd.crosstab(loan['Dependents'],loan['Loan_Status'])
Education=pd.crosstab(loan['Education'],loan['Loan_Status'])
Self_Employed=pd.crosstab(loan['Self_Employed'],loan['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4), color=['lightblue', 'lightpink'])
plt.show()
print()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.show()
print()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4), color=['lightblue', 'lightpink'])
plt.show()
print()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4), color=['lightblue', 'lightpink'])
plt.show()

Credit_History=pd.crosstab(loan['Credit_History'],loan['Loan_Status'])
Property_Area=pd.crosstab(loan['Property_Area'],loan['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(4,4), color=['lightblue', 'lightpink'])
plt.show()
print()
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.show()

### Numerical variable v target variable
loan.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar(color=['lightblue', 'lightpink'])

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
loan['Income_bin']=pd.cut(loan['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(loan['Income_bin'],loan['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

bins=[0,1000,3000,42000]
group=['Low','Average','High']
loan['Coapplicant_Income_bin']=pd.cut(loan['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(loan['Coapplicant_Income_bin'],loan['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')

loan['Total_Income']=loan['ApplicantIncome']+loan['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
loan['Total_Income_bin']=pd.cut(loan['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(loan['Total_Income_bin'],loan['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

bins=[0,100,200,700]
group=['Low','Average','High']
loan['LoanAmount_bin']=pd.cut(loan['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(loan['LoanAmount_bin'],loan['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, color=['lightblue', 'lightpink'])
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

loan=loan.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
loan['Dependents'].replace('3+', 3,inplace=True)
loan['Loan_Status'].replace('N', 0,inplace=True)
loan['Loan_Status'].replace('Y', 1,inplace=True)

matrix = loan.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap='YlGnBu');

####### MISSING VALUE AND OUTLIER TREATMENT #######
# Missing value imputation
loan.isnull().sum()

loan['Gender'].fillna(loan['Gender'].mode()[0], inplace=True)
loan['Married'].fillna(loan['Married'].mode()[0], inplace=True)
loan['Dependents'].fillna(loan['Dependents'].mode()[0], inplace=True)
loan['Self_Employed'].fillna(loan['Self_Employed'].mode()[0], inplace=True)
loan['Credit_History'].fillna(loan['Credit_History'].mode()[0], inplace=True)

loan['Loan_Amount_Term'].value_counts()

loan['Loan_Amount_Term'].fillna(loan['Loan_Amount_Term'].mode()[0], inplace=True)

loan['LoanAmount'].fillna(loan['LoanAmount'].median(), inplace=True)

loan.isnull().sum()

# Outlier Treatment
loan['LoanAmount_log'] = np.log(loan['LoanAmount'])
loan['LoanAmount_log'].hist(bins=20, color='lightblue')

####### FEATURE ENGINEERING #######
loan['Total_Income']=loan['ApplicantIncome']+loan['CoapplicantIncome']

sns.distplot(loan['Total_Income']);

loan['Total_Income_log'] = np.log(loan['Total_Income'])
sns.distplot(loan['Total_Income_log']);

loan['EMI']=loan['LoanAmount']/loan['Loan_Amount_Term']

sns.distplot(loan['EMI']);

loan['Balance Income']=loan['Total_Income']-(loan['EMI']*1000) # Multiply with 1000 to make the units equal

sns.distplot(loan['Balance Income']);

loan=loan.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Loan_ID'], axis=1)

####### CLASSIFICATION ALGORITHMS #######
X = loan.drop('Loan_Status',1)
y = loan.Loan_Status # Save target variable in separate dataset

X=pd.get_dummies(X)
loan=pd.get_dummies(loan)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3)

# Model initialization
models = [
    ('Logistic Regression', LogisticRegression()),
     ('K-Nearest Neighbors', KNeighborsClassifier()),
      ('Support Vector Machines', SVC()),
       ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
         ('AdaBoost', AdaBoostClassifier()),
          ('Gradient Boosting', GradientBoostingClassifier()),
           ('Naive Bayes', GaussianNB()),
            ('Neural Network', MLPClassifier())
]

# Function for model evaluation
def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  return accuracy, precision, recall, f1

# Training and evaluating models
results = []
for name, model in models:
  model.fit(X_train, y_train)
  accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
  results.append([name, accuracy, precision, recall, f1])
  
# Displays the results in a comparison table
results_loan = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
display(results_loan)

# Visualization of results in the form of a bar plot
custom_palette = sns.color_palette("pastel")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_loan, palette=custom_palette)
plt.title('Classification Model Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.show()


####### END #######



