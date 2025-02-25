---
title: "pedestrianFatalsFARS"
output:
  html_document:
    df_print: paged
---

2025-02-20

Author: Faline Rezvani

This R Notebook demonstrates how Python code from a Jupyter notebook can be brought into R Studio to knit a document.

Using the National Highway Traffic Safety Administration (NHTSA) Fatality Analysis Reporting System (FARS), we will inspect motor vehicle crash characteristics related to increased pedestrian risk in the U.S.

FARS data can be accessed [here](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars).

```{r setup, include=TRUE}
# Loading libraries
library(reticulate)
library(tidyverse)
```

# Installing Python packages into R Studio environment

```{r}
py_install("pandas")
py_install("numpy")
py_install("matplotlib")
py_install("scikit-learn")
py_install("imblearn")
py_install("statsmodels")
```

# Importing Python packages in Python cell

```{python, include = TRUE}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
```

# Loading FARS Auxiliary Accident and Vehicle .csv files from local location

```{python}
aux_acc = pd.read_csv('ACC_AUX22.CSV')

aux_veh = pd.read_csv('VEH_AUX22.CSV')
```

```{python}
# Checking for missing values
aux_acc.isna().sum()
```

```{python}
# Dropping column, 'CENSUS_2020_TRACT_FIPS' with 257 missing values
aux_acc = aux_acc.drop(['CENSUS_2020_TRACT_FIPS'], axis=1)
```

```{python}
# Checking for missing values
aux_veh.isna().sum()
```

```{python}
# Reducing accident and vehicle dataset correlated features
aux_acc = aux_acc.drop(['YEAR', 'A_PED', 'A_PEDAL', 'A_D15_19', 'A_D16_19', 'A_D15_20', 'A_D16_24'], axis=1)

aux_veh = aux_veh.drop(['STATE', 'YEAR', 'A_DRDIS', 'A_DRDRO', 'A_SPVEH'], axis=1)
```

# Joining Accident and Vehicle datasets

```{python}
# Using the Pandas merge function to inner join accident and vehicle tables on 'ST_CASE'
df = pd.merge(left=aux_acc, right=aux_veh, left_on='ST_CASE', right_on='ST_CASE')

# Find and drop duplicate rows
df.drop_duplicates(subset='ST_CASE', keep='first', inplace=True, ignore_index=False)
```

# A_PED_F '1' - involving pedestrian fatality; '2' - not involving pedestrian fatality

```{python}
df['A_PED_F'].value_counts()
```

```{python}
# Shuffling DataFrame, so data model doesn't learn from possible data entry pattern
df = shuffle(df)
```

Theoretically, we can represent the discovery that speeding was a factor in a crash as the outcome, or response, making ‘speeding’, or ‘not speeding’ our binary logistic variable.  The weights of predictors, or coefficient estimates, on a response will help us evaluate relationships within FARS.


```{python}
# Create instance of label encoder
le = LabelEncoder()

# Fit to target variable, 'A_SPCRA' (label)
df['A_SPCRA'] = le.fit_transform(df[['A_SPCRA']])
```

# A_SPCRA '1' - not involving speeding, '0' - involving speeding

```{python}
# Counting instances of our dependent (target) variable, 'A_SPCRA'.
df['A_SPCRA'].value_counts()
```

```{python}
# Creating .csv file from the joined accident and vehicle DataFrame
# 39,221 samples, 49 features
df.to_csv('FARS_aux_acc_veh.csv', index=False)
```

# Implementing Statistical Logit Model

```{python}
y = df['A_SPCRA']
X = df.drop(['A_SPCRA'], axis=1)
```

```{python}
# Balancing the dataset, ensuring algorithm will not be skewed
# by abundance of '1', incidents not involving speeding.
# Summarize class distribution
print(Counter(y))
```

```{python}
# Define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
```

```{python}
# Fit and apply the transform
X,y = oversample.fit_resample(X, y)
```

```{python}
# Check balance
print(Counter(y))
```

# Test/Train Split

```{python}
# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape); print(X_test.shape)
```

```{python}
# Using the statsmodels Logit function to calculating coefficients of predictors
logit1 = sm.Logit(y_train, X_train)

result1 = logit1.fit()
result1.summary()
```

Pseudo R-squared measures the variance of the error term (variance of errors), a residual variable representing the margin of error resulting from differences in a statistical model’s theoretical values and actual observed values.

The pseudo R-squared value, 0.14 of our logit model tells us that there are many other independent variables to take into consideration for understanding something so complex as a driver speeding while operating a motor vehicle.

A p-value of < the significance level of 0.05 for predictor, ‘A_PED_F’ supports evidence to reject the null hypothesis: here is no change in incidents involving speeding with a change in incidents of pedestrian fatality.

# Building Supervised Machine Learning Logistic Regression Model

```{python}
# Normalizing the independent variables
scaler_m = MinMaxScaler()

X = scaler_m.fit_transform(X.values)
```

```{python}
# Creating instance of Logistic Regression
lr = LogisticRegression(max_iter = 2000,
                        class_weight = 'balanced',
                        solver = 'saga',
                        C = 0.2)
```

```{python}
# Passing the training dataset into the model
lr.fit(X_train, y_train)
```

# Making predictions on the test dataset

```{python}
y_pred = lr.predict(X_test)
```

```{python}
# Calculating predicted probabilities of each fatality involving speeding
y_pred_prob = lr.predict_proba(X_test)
```

```{python}
# Computing accuracy
print('The accuracy score of our regression model is:', lr.score(X_test, y_test))
```

# Known vs predicted

```{python}
# Passing labeled values from test dataset and the values the machine has predicted
# Total is total observations from test dataset
cm = confusion_matrix(y_test, y_pred)
cm
```

```{python}
# Classification report for test set
print(classification_report(y_test,y_pred))
```

```{python}
# Visualizing true positive and false positive rate
lr_roc_auc = roc_auc_score(y_test, lr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.figure(figsize = (8, 6))
plt.plot(fpr, tpr, label = 'Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.savefig('ROC_lr_crashcat')
plt.show()
```


References:
[National Highway Traffic Safety Administration (NHTSA). (2021, February). Fatality Analysis Reporting System (FARS) auxiliary datasets analytical user’s manual, 1982-2019 (Report No. DOT HS 813 071). Washington, DC: Author.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813071)

