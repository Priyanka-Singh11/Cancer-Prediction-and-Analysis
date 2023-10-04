#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib','inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer=load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[7]:


cancer['feature_names']


# In[8]:


df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()


# In[9]:


cancer['target']


# In[10]:


df_target=pd.DataFrame(cancer['target'],columns=['Cancer'])


# In[11]:


df_feat.head()


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler=StandardScaler()


# In[15]:


scaler.fit(df_feat)


# In[16]:


scaler.fit(df_feat)


# In[18]:


scaler_features=scaler.transform(df_feat)


# In[20]:


df_feat_scaled=pd.DataFrame(scaler_features,columns=df_feat.columns)


# In[21]:


df_feat_scaled.head()


# In[23]:


from sklearn.model_selection import train_test_split
x_train, x_test ,y_train ,y_test =train_test_split(scaler_features, np.ravel(df_target),test_size=0.30,random_state=105)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# In[25]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[26]:


knn.fit(x_train,y_train)


# In[28]:


pred=knn.predict(x_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix


# In[31]:


print(confusion_matrix(y_test,pred))


# In[32]:


print(classification_report(y_test,pred))


# In[36]:


#choosing a k value
error_rate=[]
for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    error_rate.append(np.mean(pred_i !=y_test))


# In[43]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed' , marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[56]:


# with k=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[57]:


# with k=21
knn=KNeighborsClassifier(n_neighbors=21)
knn.fit(x_train,y_train)
pred = knn.predict(x_test)
print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




