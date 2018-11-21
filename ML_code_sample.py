
# coding: utf-8

# In[2]:


#The code for final result


# In[1]:


#import modules

get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[2]:


#read files

data=pd.read_csv('transactions.csv')
ptypes=pd.read_csv('ps_type.csv')


# In[3]:


#Feature extraction and feature engeneering

#making dictionaries to map categorical variables
lin=list(range(0,60))
local=list(zip(data['locale'].unique(),lin))
dict12=dict(local)
lin1=list(range(0,27))

#extracting some user agent info
splt1=list(zip(data['user_agent'].dropna().apply(lambda x: x[x.find("(")+1:x.find(")")].split()[0].strip(';')).unique(),lin1))
splt1=dict(splt1)
#create day-time feature
data['date_create']=pd.to_datetime(data['date_create']).dt.hour

#create categorical variables and some feature engeneering
data['splt1']=data['user_agent'].dropna().apply(lambda x: x[x.find("(")+1:x.find(")")].split()[0].strip(';'))
data['splt2']=data['splt1'].dropna().apply(lambda x: splt1[x])
data['locale1']=data['locale'].apply(lambda x: dict12[x])
data['timezone_offset'].fillna(0,inplace=True)

data['id_proj_cat']=3
data['id_proj_cat'][data['id_project']==5450]=0
data['id_proj_cat'][data['id_project']==15174]=1
data['id_proj_cat'][data['id_project']==18096]=2


# In[4]:


## choose features vector and target values

Xy=pd.merge(data, ptypes.drop_duplicates(subset=['id_instance']), how='left', left_on='id_instance',right_on='id_instance')
Xy1=pd.merge(data, ptypes.drop_duplicates(subset=['id_instance']), how='left', left_on='id_instance',right_on='id_instance')

#select first 300k objects for train and test
Xy=Xy.dropna(subset=['id_class']).iloc[:300000]

#select last 100k for final validation in order check for possible data leakage and overal generalisation perfomance
Xy1=Xy1.dropna(subset=['id_class']).iloc[700000:800000]

#features for train and test
X=Xy[['sum_dollars','currency_payment','id_proj_cat','id_bank_contr','id_country','timezone_offset']].copy()

#features for validation
X1=Xy1[['sum_dollars','currency_payment','id_proj_cat','id_bank_contr','id_country','timezone_offset']].copy()

#target values for test and validation

y=Xy['id_class']
y1=Xy1['id_class']

#create train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[5]:


#scaling
scaler=MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# In[7]:


#create classifiers
knn = KNeighborsClassifier(n_neighbors = 5)
clf2 = DecisionTreeClassifier(max_depth=16, min_samples_split=150)
clf4 = RandomForestClassifier()
clf5 = GradientBoostingClassifier()


# In[8]:


#try different models without feature preprocessing
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, y_train)
print('Knn scores train/ test',(knn.score(X_train,y_train),knn.score(X_test,y_test)))

clf2 = DecisionTreeClassifier(max_depth=16, min_samples_split=150).fit(X_train, y_train)
print('Decision trees scores train/ test',(clf2.score(X_train,y_train),clf2.score(X_test,y_test)))



clf4 = RandomForestClassifier().fit(X_train, y_train)
print('Random forests scores train/ test', (clf4.score(X_train,y_train),clf4.score(X_test,y_test)))


clf5 = GradientBoostingClassifier().fit(X_train,y_train)
print('Gradient boost scores train/ test', (clf5.score(X_train,y_train),clf5.score(X_test,y_test)))
print('')
print('Feature importance for our models:')
#print(clf2.feature_importances_)
#print(clf4.feature_importances_)
#print(clf5.feature_importances_)


fi1=clf2.feature_importances_
fi2=pd.Series(clf4.feature_importances_)
fi3=pd.Series(clf5.feature_importances_)

stck=np.stack((fi1,fi2,fi3))
features=pd.DataFrame(stck,columns=['sum_dollars','currency_payment','id_proj_cat','id_bank_contr','id_country','timezone_offset'])
pd.options.display.float_format='{:,.2f}'.format
features['Model']=['Decision tree','Random forest','Gradient boost']
features.set_index('Model', inplace=True)
features


# In[8]:


#try different models with MIN MAX scaling

knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train_scaled, y_train)
print('Knn scores train/ test',(knn.score(X_train_scaled,y_train),knn.score(X_test_scaled,y_test)))


clf4 = RandomForestClassifier().fit(X_train_scaled, y_train)
print('Random forests scores train/ test', (clf4.score(X_train_scaled,y_train),clf4.score(X_test_scaled,y_test)))


clf5 = GradientBoostingClassifier().fit(X_train_scaled,y_train)
print('Gradient boost scores train/ test', (clf5.score(X_train_scaled,y_train),clf5.score(X_test_scaled,y_test)))

#in a result we can see, that minmax scaler is usefull in this case


# In[9]:


#we selected a Decision tree classifier with max_depth=16, min_samples_split=150
clf2 =  DecisionTreeClassifier(max_depth=16, min_samples_split=150).fit(X_train, y_train)
print('Random forests scores train/ test', (clf2.score(X_train,y_train),clf2.score(X_test,y_test)))
print('Decision trees scores train/ test',(clf2.score(X1,y1)))


# In[10]:


#check how our model perfomance versus Dummy
from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

print('Dummy - most frequent train/ test', (dummy_majority.score(X_train,y_train),dummy_majority.score(X_test,y_test)))
print('Dummy validation',(dummy_majority.score(X1,y1)))

dummy_majority = DummyClassifier(strategy = 'stratified').fit(X_train, y_train)

print('Dummy stratified train/ test', (dummy_majority.score(X_train,y_train),dummy_majority.score(X_test,y_test)))
print('Dummy stratified validation train/ test',(dummy_majority.score(X1,y1)))


# In[11]:


#test our models on final validation data

print('Knn scores train/ test',(knn.score(X1,y1)))

print('Decision trees scores train/ test',(clf2.score(X1,y1)))

print('Random forests scores train/ test', (clf4.score(X1,y1)))


# In[10]:


#cross validation for decision tree

kf=KFold(n_splits=5, random_state=42, shuffle=True)

score=np.mean(cross_val_score(clf2, X, y, cv=kf, scoring='accuracy'))
print('Cross validation mean score',score)
print('core on validation data',clf2.score(X1,y1))


# In[12]:


#investigate the difference between models

preknn=clf5.predict_proba(X_test)
knn_rank=pd.DataFrame(preknn, columns=['Оплата наличными','Электронные способы оплаты','Оплата с банковского счета',                                    'Мобильные платежи','Предоплаченные карты','Оплата банковской картой'])

knn_rank['pred']=clf5.predict(X_test)

pred1=pd.Series(knn.predict(X_test))
pred2=pd.Series(clf2.predict(X_test))
pred3=pd.Series(clf4.predict(X_test))
pred4=pd.Series(clf5.predict(X_test))
rank_ensemble=pd.DataFrame(columns=['knn','Decision trees','random_for','gradient_boost'])
rank_ensemble['knn']=pred1
rank_ensemble['Decision trees']=pred2
rank_ensemble['random_for']=pred3
rank_ensemble['gradient_boost']=pred4

mode=rank_ensemble.mode(axis=1)
mode.dropna().shape
rank_ensemble.head()


# In[13]:


#get the ranks, using predict proba_method (using X_test for prediction and decision tree model)

pre=clf2.predict_proba(X_test)
df3=pd.DataFrame(pre, columns=['Оплата наличными','Электронные способы оплаты','Оплата с банковского счета',                                    'Мобильные платежи','Предоплаченные карты','Оплата банковской картой'])

#df3['return']=df3.idxmax(axis=1)
#create mapping and dataframe with results
arr=np.argsort(-df3.values,axis=1)
df4=pd.DataFrame(df3.columns[arr], index=df3.index)
#df3.iloc[13]
#df3.head()

#y_test.reset_index()
df5=pd.DataFrame( y_test.reset_index()['id_class'])
df5['pred']=clf4.predict(X_test)
df5['accuracy']=df5['id_class']==df5['pred']
df5['accuracy'].value_counts(1)
df3.head()


# In[14]:


df4['result']=df4.values.tolist()


# In[15]:


#show results on X_test (importance decreases from left to right)
#df4.iloc[13]

df4.head()


# In[16]:


#check that we dont doubled the results list
df4['result'].iloc[0]


# In[17]:


#PLOTTING CONFUSION MATRIX TO CHECK THE RESULTS
svm_predicted_mc = clf2.predict(X_test)
confusion_mc = confusion_matrix(y_test,svm_predicted_mc)
df_cm = pd.DataFrame(confusion_mc, index = [i for i in range(1,7)],
                  columns = [i for i in range(1,7)])

plt.figure(figsize = (5.5,4))
sns.heatmap(df_cm, annot=True,  fmt='g')
plt.title('Decision tree confusion matrix \nAccuracy:{0:.3f}'.format(accuracy_score(y_test,
                                                                    svm_predicted_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

df_cm



# In[18]:


#номер клиента из выборки X_test и проранжированные для него системы оплаты
df4['result']


# # Plot decision tree

# In[66]:


#in this section we will try to visualise a decision tree, but with max depth=3 in order to preserve
#simplicity in interpritation (in our ranking model we used max depth =16).
#This section is just for visualisation example of interpretability  of decision tree models
#You need to install adspy_shared_utililies and graphviz to run this section

import os
os.environ["PATH"] += os.pathsep + r'C:\Users\Alexey\Desktop\Data_science\TEST!\bin'
from adspy_shared_utilities import plot_decision_tree
from sklearn import tree

X_names=['sum_dollars','currency_payment','id_proj_cat','id_bank_contr','id_country','timezone_offset']
y_names=['cash','e-pay','bank account',                                    'Mobile','prepaid cards','credit card']

clf2 =  DecisionTreeClassifier(max_depth=3, min_samples_split=150).fit(X_train, y_train)
plot_decision_tree(clf2, X_names, y_names)
