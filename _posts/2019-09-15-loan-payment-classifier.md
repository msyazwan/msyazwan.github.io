---
title: "Finding the Best Classification Algorithm for Predicting Loan Payment"
category: [project]
tags: [machine learning, classification, supervised learning, scikit-learn]
author_profile: true
---

This project will be focussing on finding the best classifier to predict whether a loan case will be paid off or not. We will use machine learning packages from scikit-learn such as KNN, Decision Tree, SVM, and Logistic Regression.

_Credit: IBM Cognitive Class_

<img src="https://www.engineeringbigdata.com/wp-content/uploads/boston-dataset-scikit-learn-machine-learning-python-tutorial.png" width="500">

### About Dataset

This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off or defaulted                                               |
| Principal      | Basic principal loan amount                                                           |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |

### Importing Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Training Set


```python
train_url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv'
```


```python
df = pd.read_csv(train_url)
```

First off, let's view the data.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>loan_status</th>
      <th>Principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>45</td>
      <td>High School or Below</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>33</td>
      <td>Bechalor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>15</td>
      <td>9/8/2016</td>
      <td>9/22/2016</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>6</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (346, 10)




```python
df.count()
```




    Unnamed: 0        346
    Unnamed: 0.1      346
    loan_status       346
    Principal         346
    terms             346
    effective_date    346
    due_date          346
    age               346
    education         346
    Gender            346
    dtype: int64




```python
# Checking missing values
df.isnull().sum()
```




    Unnamed: 0        0
    Unnamed: 0.1      0
    loan_status       0
    Principal         0
    terms             0
    effective_date    0
    due_date          0
    age               0
    education         0
    Gender            0
    dtype: int64




```python
df.dtypes
```




    Unnamed: 0         int64
    Unnamed: 0.1       int64
    loan_status       object
    Principal          int64
    terms              int64
    effective_date    object
    due_date          object
    age                int64
    education         object
    Gender            object
    dtype: object



The data contains 346 rows and 10 columns with no missing values. The dataset were also mixed with numbers and strings.

### Data cleaning


```python
# Drop Insignificant Column
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
      <th>Principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>45</td>
      <td>High School or Below</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>33</td>
      <td>Bechalor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>15</td>
      <td>9/8/2016</td>
      <td>9/22/2016</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Renaming Column
df.rename(columns={'Principal': 'principal', "Gender": "gender"}, inplace = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
      <th>principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>45</td>
      <td>High School or Below</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>33</td>
      <td>Bechalor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>15</td>
      <td>9/8/2016</td>
      <td>9/22/2016</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Standardizing Text & Fixing Typos
print(df['loan_status'].unique())
print(df['education'].unique())
print(df['gender'].unique())
```

    ['PAIDOFF' 'COLLECTION']
    ['High School or Below' 'Bechalor' 'college' 'Master or Above']
    ['male' 'female']



```python
df['loan_status'] = df['loan_status'].apply(lambda x: 'paid_off' if (x == 'PAIDOFF')  else 'defaulted')

df.loc[df.education == 'High School or Below', 'education'] = 'high_school_or_below'
df.loc[df.education == 'college', 'education'] = 'college'
df.loc[df.education == 'Bechalor', 'education'] = 'bachelor'
df.loc[df.education == 'Master or Above', 'education'] = 'master_or_above'

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
      <th>principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>45</td>
      <td>high_school_or_below</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>33</td>
      <td>bachelor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>15</td>
      <td>9/8/2016</td>
      <td>9/22/2016</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>9/9/2016</td>
      <td>10/8/2016</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
      <th>principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-08</td>
      <td>2016-10-07</td>
      <td>45</td>
      <td>high_school_or_below</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-08</td>
      <td>2016-10-07</td>
      <td>33</td>
      <td>bachelor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>15</td>
      <td>2016-09-08</td>
      <td>2016-09-22</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-09</td>
      <td>2016-10-08</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-09</td>
      <td>2016-10-08</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Extraction

Let's try to dig information from effective date and due date, and their relationship to loan payment.

We convert the date to day of week. It will start from 0 as Monday until 6, which is Sunday.


```python
df['dayofweek_getloan'] = df['effective_date'].dt.dayofweek
df['dayofweek_dueloan'] = df['due_date'].dt.dayofweek
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_status</th>
      <th>principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>gender</th>
      <th>dayofweek_getloan</th>
      <th>dayofweek_dueloan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-08</td>
      <td>2016-10-07</td>
      <td>45</td>
      <td>high_school_or_below</td>
      <td>male</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-08</td>
      <td>2016-10-07</td>
      <td>33</td>
      <td>bachelor</td>
      <td>female</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>15</td>
      <td>2016-09-08</td>
      <td>2016-09-22</td>
      <td>27</td>
      <td>college</td>
      <td>male</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-09</td>
      <td>2016-10-08</td>
      <td>28</td>
      <td>college</td>
      <td>female</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>paid_off</td>
      <td>1000</td>
      <td>30</td>
      <td>2016-09-09</td>
      <td>2016-10-08</td>
      <td>29</td>
      <td>college</td>
      <td>male</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
bins = np.linspace(df.dayofweek_getloan.min(), df.dayofweek_getloan.max(), 10)
g = sns.FacetGrid(df, col = "gender", hue="loan_status", palette="Set1", col_wrap = 2)
g.map(plt.hist, 'dayofweek_getloan', bins = bins, ec = "k")
g.axes[-1].legend()
plt.show()
```


![png](output_27_0.png)


The charts above show that people who get their loan during the weekend (Fri-Sun) tend to not paying their loan.


```python
bins = np.linspace(df.dayofweek_dueloan.min(), df.dayofweek_dueloan.max(), 10)
g = sns.FacetGrid(df, col = "gender", hue="loan_status", palette="Set1", col_wrap = 2)
g.map(plt.hist, 'dayofweek_dueloan', bins = bins, ec = "k")
g.axes[-1].legend()
plt.show()
```


![png](output_29_0.png)


While for the due date loan charts it show that the defaulted percentage is higher on Monday and Sunday.

### Data Preprocessing

Based on the information above, we will encode the weekend (Fri to Sun) of __effective_date__ to 1, and the others to 0.

For __due_date__, we will encode Monday and Sunday to 1, and the others to 0.

Other categorical features will be encoded to 0 and 1 as well.


```python
#encode effective_date weekend
df['weekend_getloan'] = df['dayofweek_getloan'].apply(lambda x: 1 if (x > 3)  else 0)

#encode monday and sunday of due_date
df['startendweek_dueloan'] = df['dayofweek_dueloan'].apply(lambda x: 1 if (x == 0 or x == 6)  else 0)

#encode gender
gender_dummy = pd.get_dummies(df.gender)

#encode education
edu_dummy = pd.get_dummies(df.education)

#combined all new encoded features to dataframe
df = pd.concat([df, gender_dummy, edu_dummy], axis = 1)

#encode loan_status
df['loan_stat'] = df['loan_status'].apply(lambda x: 1 if (x == 'paid_off')  else 0)

#remove unused column.
df.drop(['loan_status','effective_date', 'due_date', 'dayofweek_getloan', 'dayofweek_dueloan', 'education','gender'], axis = 1, inplace = True)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>principal</th>
      <th>terms</th>
      <th>age</th>
      <th>weekend_getloan</th>
      <th>startendweek_dueloan</th>
      <th>female</th>
      <th>male</th>
      <th>bachelor</th>
      <th>college</th>
      <th>high_school_or_below</th>
      <th>master_or_above</th>
      <th>loan_stat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>30</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000</td>
      <td>30</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>15</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000</td>
      <td>30</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000</td>
      <td>30</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's get an overview on how the parameters relate to one another using correlation matrix


```python
plt.figure(figsize=(10, 5))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
```


![png](output_35_0.png)


### Feature Selection


```python
x = df.iloc[:,:-1].values
x
```




    array([[1000,   30,   45, ...,    0,    1,    0],
           [1000,   30,   33, ...,    0,    0,    0],
           [1000,   15,   27, ...,    1,    0,    0],
           ...,
           [ 800,   15,   39, ...,    1,    0,    0],
           [1000,   30,   28, ...,    1,    0,    0],
           [1000,   30,   26, ...,    1,    0,    0]])




```python
y = df.iloc[:,-1].values
y
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



### Split Train and Validation Set


```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 10)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Validation set:', x_val.shape,  y_val.shape)
```

    Train set: (242, 11) (242,)
    Validation set: (104, 11) (104,)


### Data Standardization


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
```

    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/utils/validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/utils/validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)
    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/utils/validation.py:595: DataConversionWarning: Data with input dtype int64 was converted to float64 by StandardScaler.
      warnings.warn(msg, DataConversionWarning)


### Algorithm 1: Logistic Regression


```python
from sklearn.linear_model import LogisticRegression

classifier1 = LogisticRegression(solver = 'liblinear', random_state = 0)
classifier1.fit(x_train, y_train)

y_pred = classifier1.predict(x_val)
```


```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(f1_score(y_val, y_pred))
```

    0.8076923076923077
    [[ 5 17]
     [ 3 79]]
                  precision    recall  f1-score   support
    
               0       0.62      0.23      0.33        22
               1       0.82      0.96      0.89        82
    
       micro avg       0.81      0.81      0.81       104
       macro avg       0.72      0.60      0.61       104
    weighted avg       0.78      0.81      0.77       104
    
    0.8876404494382022


### Algorithm 2: Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

classifier2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier2.fit(x_train, y_train)

y_pred = classifier2.predict(x_val)
```


```python
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(f1_score(y_val, y_pred))
```

    0.7019230769230769
    [[ 9 13]
     [18 64]]
                  precision    recall  f1-score   support
    
               0       0.33      0.41      0.37        22
               1       0.83      0.78      0.81        82
    
       micro avg       0.70      0.70      0.70       104
       macro avg       0.58      0.59      0.59       104
    weighted avg       0.73      0.70      0.71       104
    
    0.8050314465408805


### Algorithm 3: Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier

classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier3.fit(x_train, y_train)

y_pred = classifier3.predict(x_val)
```


```python
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(f1_score(y_val, y_pred))
```

    0.6730769230769231
    [[ 9 13]
     [21 61]]
                  precision    recall  f1-score   support
    
               0       0.30      0.41      0.35        22
               1       0.82      0.74      0.78        82
    
       micro avg       0.67      0.67      0.67       104
       macro avg       0.56      0.58      0.56       104
    weighted avg       0.71      0.67      0.69       104
    
    0.7820512820512819


### Algorithm 4: K-Nearest Neighbour


```python
from sklearn.tree import DecisionTreeClassifier

classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier3.fit(x_train, y_train)

y_pred = classifier3.predict(x_val)
```


```python
print(accuracy_score(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(f1_score(y_val, y_pred))
```

    0.6730769230769231
    [[ 9 13]
     [21 61]]
                  precision    recall  f1-score   support
    
               0       0.30      0.41      0.35        22
               1       0.82      0.74      0.78        82
    
       micro avg       0.67      0.67      0.67       104
       macro avg       0.56      0.58      0.56       104
    weighted avg       0.71      0.67      0.69       104
    
    0.7820512820512819


### Evaluating Test Set


```python
test_url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv'
```


```python
df_test = pd.read_csv(test_url)
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>loan_status</th>
      <th>Principal</th>
      <th>terms</th>
      <th>effective_date</th>
      <th>due_date</th>
      <th>age</th>
      <th>education</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/8/2016</td>
      <td>10/7/2016</td>
      <td>50</td>
      <td>Bechalor</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>5</td>
      <td>PAIDOFF</td>
      <td>300</td>
      <td>7</td>
      <td>9/9/2016</td>
      <td>9/15/2016</td>
      <td>35</td>
      <td>Master or Above</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21</td>
      <td>21</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/10/2016</td>
      <td>10/9/2016</td>
      <td>43</td>
      <td>High School or Below</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24</td>
      <td>24</td>
      <td>PAIDOFF</td>
      <td>1000</td>
      <td>30</td>
      <td>9/10/2016</td>
      <td>10/9/2016</td>
      <td>26</td>
      <td>college</td>
      <td>male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>35</td>
      <td>PAIDOFF</td>
      <td>800</td>
      <td>15</td>
      <td>9/11/2016</td>
      <td>9/25/2016</td>
      <td>29</td>
      <td>Bechalor</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



### Preprocess Test Set


```python
# Drop Insignificant Column
df_test.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace = True)

#Renaming Column
df_test.rename(columns={'Principal': 'principal', "Gender": "gender"}, inplace = True)

# Standardizing Text & Fixing Typos
df_test['loan_status'] = df_test['loan_status'].apply(lambda x: 'paid_off' if (x == 'PAIDOFF')  else 'defaulted')

df_test.loc[df_test.education == 'High School or Below', 'education'] = 'high_school_or_below'
df_test.loc[df_test.education == 'college', 'education'] = 'college'
df_test.loc[df_test.education == 'Bechalor', 'education'] = 'bachelor'
df_test.loc[df_test.education == 'Master or Above', 'education'] = 'master_or_above'

# Convert to date time object
df_test['due_date'] = pd.to_datetime(df_test['due_date'])
df_test['effective_date'] = pd.to_datetime(df_test['effective_date'])

df_test['dayofweek_getloan'] = df_test['effective_date'].dt.dayofweek
df_test['dayofweek_dueloan'] = df_test['due_date'].dt.dayofweek

#encode effective_date weekend
df_test['weekend_getloan'] = df_test['dayofweek_getloan'].apply(lambda x: 1 if (x > 3)  else 0)

#encode monday and sunday of due_date
df_test['startendweek_dueloan'] = df_test['dayofweek_dueloan'].apply(lambda x: 1 if (x == 0 or x == 6)  else 0)

#encode gender
gender_dummy = pd.get_dummies(df_test.gender)

#encode education
edu_dummy = pd.get_dummies(df_test.education)

#combined all new encoded features to dataframe
df_test = pd.concat([df_test, gender_dummy, edu_dummy], axis = 1)

#encode loan_status
df_test['loan_stat'] = df_test['loan_status'].apply(lambda x: 1 if (x == 'paid_off')  else 0)

#remove unused column.
df_test.drop(['loan_status','effective_date', 'due_date', 'dayofweek_getloan', 'dayofweek_dueloan', 'education','gender'], axis = 1, inplace = True)

df_test.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>principal</th>
      <th>terms</th>
      <th>age</th>
      <th>weekend_getloan</th>
      <th>startendweek_dueloan</th>
      <th>female</th>
      <th>male</th>
      <th>bachelor</th>
      <th>college</th>
      <th>high_school_or_below</th>
      <th>master_or_above</th>
      <th>loan_stat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>30</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300</td>
      <td>7</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>30</td>
      <td>43</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000</td>
      <td>30</td>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>800</td>
      <td>15</td>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_test = df_test.iloc[:,:-1].values
y_test = df_test.iloc[:,-1].values
```

### Logistic Regression: Test Set


```python
y_pred = classifier1.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

    0.7407407407407407
    [[ 0 14]
     [ 0 40]]
                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00        14
               1       0.74      1.00      0.85        40
    
       micro avg       0.74      0.74      0.74        54
       macro avg       0.37      0.50      0.43        54
    weighted avg       0.55      0.74      0.63        54
    
    0.851063829787234


    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /opt/conda/envs/Python36/lib/python3.6/site-packages/sklearn/metrics/classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


### Random Forest: Test Set


```python
y_pred = classifier2.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

    0.7222222222222222
    [[10  4]
     [11 29]]
                  precision    recall  f1-score   support
    
               0       0.48      0.71      0.57        14
               1       0.88      0.72      0.79        40
    
       micro avg       0.72      0.72      0.72        54
       macro avg       0.68      0.72      0.68        54
    weighted avg       0.77      0.72      0.74        54
    
    0.7945205479452054


### Decision Tree: Test Set


```python
y_pred = classifier3.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))
```

    0.7222222222222222
    [[10  4]
     [11 29]]
                  precision    recall  f1-score   support
    
               0       0.48      0.71      0.57        14
               1       0.88      0.72      0.79        40
    
       micro avg       0.72      0.72      0.72        54
       macro avg       0.68      0.72      0.68        54
    weighted avg       0.77      0.72      0.74        54
    
    0.7945205479452054



