#!/usr/bin/env python
# coding: utf-8

# In[5]:


path = '/Users/ls/Desktop/Rec_Cap_data.xlsx'
import pandas as pd
df = pd.read_excel(path)


# In[6]:


df


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[25]:


df.columns


# In[26]:


# preproc = ColumnTransformer(transformers = [('hot', OneHotEncoder(), ['Captain ID','Cargo Type','Ship Type'])])


# In[42]:


encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[['Captain ID','Cargo Type','Ship Type']])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['Captain ID','Cargo Type','Ship Type']))
df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(['Captain ID','Cargo Type','Ship Type'], axis=1)


# In[43]:


data = df_encoded.drop(['Trip ID'], axis = 1)
x = df_encoded.drop(['Prime Proba'], axis = 1)


# In[44]:


modelPipeline = (Pipeline(steps=[('classifier', RandomForestRegressor(n_estimators=100, random_state=42))]))


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x, data['Prime Proba'], test_size = 0.3, random_state = 42)


# In[46]:


modelPipeline.fit(x_train, y_train)


# In[47]:


y_pred = modelPipeline.predict(x_test)


# In[53]:


from sklearn.metrics import r2_score
r2_score(y_pred,y_test)


# In[54]:


y_tot_pred = modelPipeline.predict(x)


# In[56]:


pred = pd.DataFrame(y_tot_pred)


# In[66]:


df['Pred_Prob'] = y_tot_pred


# In[67]:


df


# In[103]:


model_class = int(input("enter class"))
print('\033[1m'+"Recommendation:")
df[df['Pred_Prob']==(df[df.loc[:,'Model Class']==model_class]['Pred_Prob'].max())].loc[:,["Captain ID", "Ship Type", "Cargo Type"]]


# In[ ]:




