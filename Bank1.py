#!/usr/bin/env python
# coding: utf-8

# # Importing Dataset

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_dt=pd.read_csv('bank.csv',delimiter=';')
train_dt.head(10)


# In[ ]:


train_dt.info()


# In[ ]:


for i in train_dt.columns:
    train_dt[i]=np.where(train_dt[i]=='unknown',np.nan,train_dt[i])
print(train_dt.isna().sum())
print((train_dt.isna().sum()/train_dt.shape[0])*100)


# # Data Visualization and data pre-processing

# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='marital',y='age',data=train_dt, palette='rainbow')
plt.title("Age by marital status")


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='age',y='y',data=train_dt, palette='rainbow')
plt.title("Age by response")


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='age',y='education',data=train_dt, palette='rainbow')
plt.title("Age by education")


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='age',y='default',data=train_dt, palette='rainbow')
plt.title("Age by default")


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='age',y='housing',data=train_dt, palette='rainbow')
plt.title("Age by housing")


# In[ ]:


plt.figure(figsize=(8,5))
sns.boxplot(x='age',y='loan',data=train_dt, palette='rainbow')
plt.title("Age by loan")


# In[ ]:


sns.displot(data=train_dt,x="age",col="marital",kind="hist",aspect=1.4)


# In[ ]:


train_dt.drop(columns='poutcome',inplace=True)


# In[ ]:


categorical_var=[]
continuous_var=[]
for i in range(len(train_dt.columns)):
    if train_dt[train_dt.columns[i]].dtype=='O':
        categorical_var.append(train_dt.columns[i])
    elif train_dt.columns[i]!='y':
        continuous_var.append(train_dt.columns[i])


# In[ ]:


print(categorical_var)
print(continuous_var)


# In[ ]:


for i in categorical_var:
    print(train_dt[i].value_counts())


# In[ ]:


train_dt['job'].unique()


# In[ ]:


train_dt.shape


# In[ ]:


for i in range(len(categorical_var)):
    sns.countplot(train_dt[categorical_var[i]])
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


plt.figure(figsize=(12,4))
sns.countplot(train_dt['day'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(12,4))
sns.countplot(train_dt['previous'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


for i in range(len(categorical_var)-1):
    sns.countplot(train_dt[categorical_var[i]],hue=train_dt["y"])
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["age"].plot.box(color='green')
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["age"],color='red')
plt.show()


# In[ ]:


fig=plt.figure(figsize=(13,6))
ax1=fig.add_subplot(1,2,1)
train_dt["balance"].plot.box()
ax2=fig.add_subplot(1,2,2)
sns.distplot(train_dt["balance"])
plt.show()


# In[ ]:


plt.figure(figsize=(200,100))
train_dt.boxplot(column="balance",by="y") 
plt.suptitle("")


# # Missing value imputation

# In[ ]:


from scipy.stats import chi2_contingency
from scipy.stats.contingency import association


# In[ ]:


for i in range(len(categorical_var)):
    for j in range((i+1),len(categorical_var)):
        table=pd.crosstab(train_dt[categorical_var[i]],train_dt[categorical_var[j]])
        stat,p,dof,expected=chi2_contingency(table)
        if p <=0.01:
            print(categorical_var[i],"and",categorical_var[j],"are dependent with association :",association(table))


# In[ ]:


sns.countplot(train_dt['job'],hue=train_dt["contact"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


train_dt['contact']=train_dt['contact'].fillna(train_dt['contact'].mode()[0])


# In[ ]:


sns.countplot(train_dt['month'],hue=train_dt["education"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


for i in list(train_dt[train_dt['education'].isna()].index):
    if (train_dt['month'][i]=='may') or (train_dt['month'][i]=='apr') or (train_dt['month'][i]=='jun') or (train_dt['month'][i]=='jul') or (train_dt['month'][i]=='jan') or (train_dt['month'][i]=='feb') or (train_dt['month'][i]=='nov') or (train_dt['month'][i]=='mar') or (train_dt['month'][i]=='oct'):
        train_dt.iloc[i,3]='secondary' 
    else:
        train_dt.iloc[i,3]='tertiary'


# In[ ]:


train_dt.isna().sum()


# In[ ]:


sns.countplot(train_dt['job'],hue=train_dt["education"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


for i in list(train_dt[train_dt['job'].isna()].index):
    if (train_dt['education'][i]=='tertiary'):
        train_dt.iloc[i,1]='management' 
    else:
        train_dt.iloc[i,1]='blue-collar'


# In[ ]:


train_dt.isna().sum()


# In[ ]:


train_dt1=pd.get_dummies(train_dt[categorical_var],drop_first=True)
train_dt1.shape


# In[ ]:


train_dt_new=pd.concat([train_dt,train_dt1],axis=1)
train_dt_new=train_dt_new.drop(categorical_var,axis=1)
train_dt_new.head()


# In[ ]:


# breaking the dataset into dependent (y) and independent (x) variables
x = train_dt_new.drop("y_yes",axis =1)
y = train_dt_new["y_yes"]


# # Checking multicollinearity and feature selection

# In[ ]:


#VIF checking
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# In[ ]:


vif_df=pd.DataFrame()
vif_df['variable']=x.columns
vif_df


# In[ ]:


vif_df['VIF']=[vif(x.values,i) for i in range(x.shape[1])]
vif_df


# In[ ]:


x.drop(['age','day','marital_married','month_may'],axis=1,inplace=True)


# In[ ]:


x.columns


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier as etc


# In[ ]:


plt.figure(figsize=(10,10))
model=etc()
model.fit(x,y)
imp_var=pd.Series(model.feature_importances_,index=x.columns)
imp_var.plot(kind="barh")
plt.show()


# In[ ]:


model=etc()
model.fit(x,y)
imp_var=pd.Series(model.feature_importances_,index=x.columns)
imp_var=imp_var.sort_values(ascending=True)
imp_var


# # Model building and model validation

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.preprocessing import MinMaxScaler


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr_accuracy1=[]
lr_precision1=[]
lr_f1score1=[]
lr_recall1=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        lr_model1=LogisticRegression()
        lr_model1.fit(x_train1,y_train1)
        pred_y1=lr_model1.predict(x_cv1)
        lr_accuracy1.append(accuracy_score(y_cv1,pred_y1))
        lr_precision1.append(precision_score(y_cv1,pred_y1))
        lr_f1score1.append(f1_score(y_cv1,pred_y1))
        lr_recall1.append(recall_score(y_cv1,pred_y1))


# In[ ]:


lr_accuracy1


# In[ ]:


lr_precision1


# In[ ]:


lr_recall1


# In[ ]:


lr_f1score1


# In[ ]:


param_grid1 = {'penalty':['l2'],'C':[0.01,0.1,1,100,1000],'class_weight':['balanced'],'solver':['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']}
grid1 = GridSearchCV(LogisticRegression(),param_grid1)
grid1.fit(x_train1,y_train1)


# In[ ]:


print(grid1.best_params_)
print(grid1.score(x_cv1,y_cv1))


# In[ ]:


sc1=MinMaxScaler()
x_train1=sc1.fit_transform(x_train1)
x_cv1=sc1.transform(x_cv1)
lr_model1=LogisticRegression(class_weight= 'balanced', penalty= 'l2', solver= 'lbfgs')
lr_model1.fit(x_train1,y_train1)
pred_y1=lr_model1.predict(x_cv1)
lr_accuracy1=accuracy_score(y_cv1,pred_y1)
lr_precision1=precision_score(y_cv1,pred_y1)
lr_f1score1=f1_score(y_cv1,pred_y1)
lr_recall1=recall_score(y_cv1,pred_y1)
print("lr_accuracy =",lr_accuracy1)
print("lr_precision =",lr_precision1)
print("lr_f1score =",lr_f1score1)
print("lr_recall =",lr_recall1)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier 


# In[ ]:


dt_accuracy2=[]
dt_precision2=[]
dt_f1score2=[]
dt_recall2=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        dtc1=DecisionTreeClassifier(random_state=2)
        dtc1.fit(x_train1,y_train1)
        pred_y1=dtc1.predict(x_cv1)
        dt_accuracy2.append(accuracy_score(y_cv1,pred_y1))
        dt_precision2.append(precision_score(y_cv1,pred_y1))
        dt_f1score2.append(f1_score(y_cv1,pred_y1))
        dt_recall2.append(recall_score(y_cv1,pred_y1))


# In[ ]:


dt_accuracy2


# In[ ]:


dt_precision2


# In[ ]:


dt_f1score2


# In[ ]:


dt_recall2


# In[ ]:


param_grid2={'criterion':['gini','entropy'],'max_depth':list(range(5,20)),'min_samples_split':list(range(2,10)),'min_samples_leaf':list(range(5,15))}
grid2 = GridSearchCV(DecisionTreeClassifier(),param_grid2)
grid2.fit(x_train1,y_train1)


# In[ ]:


print(grid2.best_params_)
print(grid2.score(x_cv1,y_cv1))


# In[ ]:


sc1=MinMaxScaler()
x_train1=sc1.fit_transform(x_train1)
x_cv1=sc1.transform(x_cv1)
dtc1=DecisionTreeClassifier(criterion= 'entropy', max_depth= 5, min_samples_leaf= 5, min_samples_split= 9)
dtc1.fit(x_train1,y_train1)
pred_y1=dtc1.predict(x_cv1)
dt_accuracy2=accuracy_score(y_cv1,pred_y1)
dt_precision2=precision_score(y_cv1,pred_y1)
dt_f1score2=f1_score(y_cv1,pred_y1)
dt_recall2=recall_score(y_cv1,pred_y1)
print("dt_accuracy =",dt_accuracy2)
print("dt_precision =",dt_precision2)
print("dt_f1score =",dt_f1score2)
print("dt_recall =",dt_recall2)


# ## Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf_accuracy3=[]
rf_precision3=[]
rf_f1score3=[]
rf_recall3=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        rf1=regressor.fit(x_train1,y_train1)
        pred_y1=regressor.predict(x_cv1)
        pred_y1_new=[]
        for i in pred_y1:
            pred_y1_new.append(1) if i>=0.5 else pred_y1_new.append(0)
        rf_accuracy3.append(accuracy_score(y_cv1,pred_y1_new))
        rf_precision3.append(precision_score(y_cv1,pred_y1_new))
        rf_f1score3.append(f1_score(y_cv1,pred_y1_new))
        rf_recall3.append(recall_score(y_cv1,pred_y1_new))


# In[ ]:


rf_accuracy3


# In[ ]:


rf_precision3


# In[ ]:


rf_f1score3


# In[ ]:


rf_recall3


# ## K-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn_accuracy4=[]
knn_precision4=[]
knn_f1score4=[]
knn_recall4=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        knn1=KNeighborsClassifier()
        knn1.fit(x_train1,y_train1)
        pred_y1=knn1.predict(x_cv1)
        knn_accuracy4.append(accuracy_score(y_cv1,pred_y1))
        knn_precision4.append(precision_score(y_cv1,pred_y1))
        knn_f1score4.append(f1_score(y_cv1,pred_y1))
        knn_recall4.append(recall_score(y_cv1,pred_y1))


# In[ ]:


knn_accuracy4


# In[ ]:


knn_precision4


# In[ ]:


knn_recall4


# In[ ]:


knn_f1score4


# ## Support Vector Machine

# In[ ]:


from sklearn import svm


# In[ ]:


svm_accuracy5=[]
svm_precision5=[]
svm_f1score5=[]
svm_recall5=[]
x3=x
for i in list(imp_var.index):
    if len(x3.columns)>8:
        x3=x3.drop(i,axis=1)
        x_train1,x_cv1,y_train1,y_cv1=train_test_split(x3,y,stratify=y,test_size=0.3,random_state=1)
        sc1=MinMaxScaler()
        x_train1=sc1.fit_transform(x_train1)
        x_cv1=sc1.transform(x_cv1)
        svm1=svm.SVC()
        svm1.fit(x_train1,y_train1)
        pred_y1=svm1.predict(x_cv1)
        svm_accuracy5.append(accuracy_score(y_cv1,pred_y1))
        svm_precision5.append(precision_score(y_cv1,pred_y1))
        svm_f1score5.append(f1_score(y_cv1,pred_y1))
        svm_recall5.append(recall_score(y_cv1,pred_y1))


# In[ ]:


svm_accuracy5


# In[ ]:


svm_precision5


# In[ ]:


svm_recall5


# In[ ]:


svm_f1score5


# In[ ]:


#svm
from sklearn.model_selection import GridSearchCV

param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6]}
grid = GridSearchCV(svm.SVC(),param_grid)
grid.fit(x_train1,y_train1)


# In[ ]:


print(grid.best_params_)
print(grid.score(x_cv1,y_cv1))


# In[ ]:


sc1=MinMaxScaler()
x_train1=sc1.fit_transform(x_train1)
x_cv1=sc1.transform(x_cv1)
svm1=svm.SVC(C= 100, degree= 1, kernel='rbf')
svm1.fit(x_train1,y_train1)
pred_y1=svm1.predict(x_cv1)
svm_accuracy5=accuracy_score(y_cv1,pred_y1)
svm_precision5=precision_score(y_cv1,pred_y1)
svm_f1score5=f1_score(y_cv1,pred_y1)
svm_recall5=recall_score(y_cv1,pred_y1)
print("svm_accuracy =",svm_accuracy5)
print("svm_precision =",svm_precision5)
print("svm_f1score =",svm_f1score5)
print("svm_recall =",svm_recall5)

