#!/usr/bin/env python
# coding: utf-8

# ## Problem Description

# A Regional Bank XYZ with 40000+ Customers would like to expand its business by predicting Customer's behavior to better sell cross products (eg: Selling Term Deposits to Retail Customers). The Bank has approached us to assess the same by providing access to their Customer campaign data. 
# 
# The data is related with direct marketing campaigns. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# Predict if an existing customer would subscribe to a Term Deposit

# #### Attribute information:

# Input variables:
# 
# 1 - age (numeric)
# 
# 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                    "blue-collar","self-employed","retired","technician","services") 
# 
# 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
# 
# 4 - education (categorical: "unknown","secondary","primary","tertiary")
# 
# 5 - default: has credit in default? (binary: "yes","no")
# 
# 6 - balance: average yearly balance, in euros (numeric) 
# 
# 7 - housing: has housing loan? (binary: "yes","no")
# 
# 8 - loan: has personal loan? (binary: "yes","no") 
# 
# ##### Related with the last contact of the current campaign:
# 
# 9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
# 
# 10 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# 11 - duration: last contact duration, in seconds (numeric)
# 
# ##### Other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
# 
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 
# 15 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
# ##### Output variable (desired target):
# 
# 16 - y - has the client subscribed a term deposit? (binary: "yes","no")
# 

# ## Load and study the data set

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, recall_score ,precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[3]:


df = pd.read_csv("Bank_Data.csv")


# In[4]:


df


# In[5]:


df.dtypes


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.y.value_counts()


# In[10]:


df.y.value_counts(normalize=True)


# In[11]:


# lets enumerate the target into 'yes' or 'no'

df['y']=df['y'].apply(lambda x: 0 if x.strip() =='no' else 1)


# In[12]:


#df['y']=df['y'].map({'yes':1,'no':0})


# In[13]:


#df['y']


# In[14]:


df.y.value_counts()


# In[15]:


cat_attrb= df.select_dtypes(include= 'object').columns
df[cat_attrb]= df[cat_attrb].astype("category")


# In[16]:


df.dtypes


# In[17]:


data=pd.get_dummies(columns= cat_attrb,data=df, prefix_sep='_',drop_first= True)
data.head()


# In[18]:


x= data.loc[ : ,data.columns.difference(['y'])]
y=data.y


# In[19]:


x.head()


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.25,random_state=2022)


# In[21]:


# STANDARDERDISING THE NUMERICAL ATTRIBUTE


# In[22]:


df.columns


# In[23]:


# scaler.fit_transform  is used  to tranfer the mean std of data
# where scaler.transfrm does not transfer mean and std(known as parameters)


# In[24]:


scaler=MinMaxScaler()
x_train[['age', 'balance','duration','pdays','previous','campaign']] = scaler.fit_transform(x_train[['age', 'balance','duration','pdays','previous','campaign']])
x_test[['age', 'balance','duration','pdays','previous','campaign']]=scaler.transform(x_test[['age', 'balance','duration','pdays','previous','campaign']])


# In[25]:


x_train=sm.add_constant(x_train)
x_test=sm.add_constant(x_test)


# In[26]:


logit_model=sm.Logit(y_train,x_train)
result= logit_model.fit()
result.summary2()


# In[27]:


thresold= 0.5       # if probability less then 0.5 then consider whole class as one 

train_pred_prob=result.predict(x_train)
train_pred= np.where(train_pred_prob > thresold, 1,0)  # if value geateer than 0.5 prints '1' and otherwise '0'


# In[28]:


train_pred_prob


# In[29]:


train_pred[:10]


# In[30]:


test_pred_prob=result.predict(x_test)
test_pred=np.where(test_pred_prob>thresold,1,0)


# In[31]:


test_pred_prob


# In[32]:


test_pred


# In[33]:


#TEST OF PREDICTION :CONFUSION MATRIX


# In[34]:


confusion_matrix(y_test,test_pred)   # y test is the prediction of x 


# In[35]:


#FINDING INFINITY ERRORS


# In[36]:


validation_accuracy=accuracy_score(y_test, test_pred)
validation_recall=recall_score(y_test,test_pred)
validation_precision=precision_score(y_test,test_pred)


# In[37]:


print('accuracy',validation_accuracy)
print('recall',validation_recall)
print('precision',validation_precision)


# In[38]:


confusion_matrix(y_train,train_pred)  # optionmsl conditions for test for train


# In[39]:


# ROC AND AUC    RECIVING OPERATOR CHARACTERSTIC CURVE


# In[40]:


fpr,tpr,threshold=roc_curve(y_train,train_pred)    # thresold is a dummyvariale here
roc_auc=auc(fpr,tpr)


# In[41]:


plt.plot([0,1],[0,1],color='navy',lw=4,linestyle='--')

plt.plot(fpr, tpr, color='orange',lw=6, label='roc curve(area=%0.2f)'%roc_auc)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')

plt.show()


# In[ ]:





# In[ ]:




