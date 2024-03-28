#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df= pd.read_excel("Dots Potential Customer Survey Data.xlsx")


# In[5]:


df.head()


# In[6]:


df.shape


# In[8]:


plt.figure(figsize=(50,15))
df.boxplot()
plt.show()


# In[9]:


#since cant see my data clearly i started with the columns that doesnot have outliers,

plt.figure(figsize=(3,2))
sns.boxplot(df['Age'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Annual Family Income ($)'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Time spent watching videos/TV'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Time spent playing indoor sports'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Time spent playing outdoor sports'])


plt.figure(figsize=(3,2))
sns.boxplot(df['Has Diabetes'])

plt.figure(figsize=(3,2))
sns.boxplot(df['Smoker'])
plt.show()


# In[10]:


plt.figure(figsize=(3,2))
sns.boxplot(df['Time spent playing outdoor sports'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['English speaker'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Migrated within country'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Migrated overseas'])
plt.show()


plt.figure(figsize=(3,2))
sns.boxplot(df['Maritial Status (0 - Single, 1 - Married, 2 - Divorced)'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Drinks alcohol'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Has debt'])


# In[11]:


plt.figure(figsize=(3,2))
sns.boxplot(df['Has Diabetes'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Has OTT subscription'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Has OTT subscription'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Likes spicy food'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df[ 'Likes desserts'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Drinks alcohol'])
plt.show()

plt.figure(figsize=(3,2))
sns.boxplot(df['Has kids'])
plt.show()


# In[12]:


import warnings # used this libary to avoid gettimg the red warning
warnings.filterwarnings("ignore")

plt.figure(figsize=(5,3))
sns.distplot(df['IQ']) #i used distplot to see the distribution of the column feature
plt.show()

plt.figure(figsize=(5,3))
sns.boxplot(df['IQ']) #i used distplot to see the outliers of the column feature
plt.show()


# In[21]:


#i used the percentile to get Q1,Q2,IQR as well as upper limit and lower limits
Q1= np.percentile(df['IQ'],25)
Q3= np.percentile(df['IQ'],75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(lower_limit,upper_limit,IQR)


# In[22]:


# i created this four loop to because i want too show the values above and below limits

selected_column = df['IQ']

for data_point in selected_column:
    if data_point < lower_limit:
        
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    
below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 


# In[23]:


#I wanted to find below limits values
print("Data points below lower limit:", below_lower_limit)


# In[24]:


#I wanted to find below limits values
print("Data points above upper limit:", above_upper_limit)


# In[25]:


df['IQ'].median()


# In[26]:


df['IQ'].mean()


# In[27]:


#im using the median to treat the outliers
median_within_range = np.median(df[(df['IQ'] >= lower_limit) & (df['IQ'] <= upper_limit)]['IQ'])
df.loc[df['IQ'] > upper_limit, 'IQ'] = median_within_range

df.loc[df['IQ'] < lower_limit, 'IQ'] = median_within_range


# # I WILL BE REPEATING SAME THING UNTILL IM DONE FIND AND TREATING OUTLIERS IN THE DATASET

# In[29]:


plt.figure(figsize=(5,3))
sns.distplot(df[ 'Total Time spent working in front of screen'])
plt.show()

plt.figure(figsize=(5,3))
sns.boxplot(df[ 'Total Time spent working in front of screen'])
plt.show()


# In[33]:


Q1= np.percentile(df['Total Time spent working in front of screen'],25)
Q3= np.percentile(df['Total Time spent working in front of screen'],75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(lower_limit,upper_limit,IQR)


# In[32]:


selected_column = df['Total Time spent working in front of screen']

for data_point in selected_column:
    if data_point < lower_limit:
        
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 


# In[34]:


print("Data points below lower limit:", below_lower_limit)


# In[35]:


print("Data points above upper limit:", above_upper_limit)


# In[36]:


df['Total Time spent working in front of screen'].mean()


# In[37]:


df['Total Time spent working in front of screen'].median()


# In[38]:


median_within_range = np.median(df[(df['Total Time spent working in front of screen'] >= lower_limit) & (df['Total Time spent working in front of screen'] <= upper_limit)]['Total Time spent working in front of screen'])
df.loc[df['Total Time spent working in front of screen'] > upper_limit, 'Total Time spent working in front of screen'] = median_within_range

df.loc[df['Total Time spent working in front of screen'] < lower_limit, 'Total Time spent working in front of screen'] = median_within_range


# In[40]:


plt.figure(figsize=(5,3))
sns.distplot(df['Sleeping hours'])
plt.show()

plt.figure(figsize=(5,3))
sns.boxplot(df['Sleeping hours'])
plt.show()


# In[41]:


Q1= np.percentile(df['Sleeping hours'],25)
Q3= np.percentile(df['Sleeping hours'],75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(lower_limit,upper_limit,IQR)


# In[42]:


selected_column = df['Sleeping hours']

for data_point in selected_column:
    if data_point < lower_limit:
      
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 


# In[43]:


print("Data points below lower limit:", below_lower_limit)


# In[44]:


print("Data points above upper limit:", above_upper_limit)


# In[45]:


df['Sleeping hours']. mean()


# In[46]:


df['Sleeping hours']. median()


# In[47]:


median_within_range = np.median(df[(df['Sleeping hours'] >= lower_limit) & (df['Sleeping hours'] <= upper_limit)]['Sleeping hours'])
df.loc[df['Sleeping hours'] > upper_limit, 'Sleeping hours'] = median_within_range

df.loc[df['Sleeping hours'] < lower_limit, 'Sleeping hours'] = median_within_range


# In[48]:


plt.figure(figsize=(5,3))
sns.distplot(df['Has Gym Subscription'])
plt.show()

plt.figure(figsize=(5,3))
sns.boxplot(df['Has Gym Subscription'])
plt.show()


# In[49]:


Q1= np.percentile(df['Has Gym Subscription'],25)
Q3= np.percentile(df['Has Gym Subscription'],75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(IQR,lower_limit,upper_limit)


# In[50]:


selected_column = df['Has Gym Subscription']

for data_point in selected_column:
    if data_point < lower_limit:
        
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 


# In[51]:


print("Data points below lower limit:", below_lower_limit)


# In[53]:


print("Data points below lower limit:",above_upper_limit )


# In[55]:


df['Has Gym Subscription'].mean()


# In[56]:


df['Has Gym Subscription'].median()


# In[ ]:





# In[ ]:


plt.figure(figsize=(5,3))
sns.distplot(df['Wants to change career'])
plt.show()

plt.figure(figsize=(5,3))
sns.boxplot(df['Wants to change career'])
plt.show()


# In[57]:


Q1= np.percentile(df['Wants to change career'],25)
Q3= np.percentile(df['Wants to change career'],75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(IQR,lower_limit,upper_limit)


# In[58]:


selected_column = df['Wants to change career']

for data_point in selected_column:
    if data_point < lower_limit:
        
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 


# In[59]:


print("Data points below lower limit:", below_lower_limit)


# In[60]:


print("Data points below lower limit:", above_upper_limit)


# In[61]:


df['Wants to change career'].mean()


# In[62]:


df['Wants to change career'].median()


# In[ ]:


#THE ARE TWO COLUMNS IN THIS DATASET WHICH ARE 'Wants to change career' AND 'Has Gym Subscription' AND THEY ARE IN BINARY VULUE (0 and 1).
#AND MY  boxplot is indicating that there's an outlier represented by the value 1,I DONT THINK IT IS necessarily TO TREAT THEM CAUSE IN THE HUMAN WORLD THIS IS A YES OR NO ANSWER AND THIS OULIERS MAKE SENSE


# In[74]:


plt.figure(figsize=(5,3))
sns.distplot(df['Number of friends'])
plt.show()

plt.figure(figsize=(5,3))
df.boxplot(column='Number of friends')
plt.show()


# In[75]:


Q1= df['Number of friends'].quantile(0.25)
Q3= df['Number of friends'].quantile(0.75)
IQR= Q3-Q1

upper_limit= Q3+ (1.5* IQR)
lower_limit= Q1- (1.5* IQR)
upper_limit,lower_limit

print(lower_limit,upper_limit,IQR)


# In[76]:


selected_column = df['Number of friends']

for data_point in selected_column:
    if data_point < lower_limit:
        
        pass
    elif data_point > upper_limit:
        
        pass
    else:
        
        pass
    
    below_lower_limit = []
above_upper_limit = []

for data_point in selected_column:
    if data_point < lower_limit:
        below_lower_limit.append(data_point)  
    elif data_point > upper_limit:
        above_upper_limit.append(data_point) 



# In[77]:


print("Data points below lower limit:", below_lower_limit)


# In[78]:


print("Data points below lower limit:",above_upper_limit )


# In[79]:


df['Number of friends'].mean()


# In[80]:


df['Number of friends'].median()


# In[81]:


median_within_range = np.median(df[(df['Number of friends'] >= lower_limit) & (df['Number of friends'] <= upper_limit)]['Number of friends'])
df.loc[df['Number of friends'] > upper_limit, 'Number of friends'] = median_within_range

df.loc[df['Number of friends'] < lower_limit, 'Number of friends'] = median_within_range


# In[82]:


plt.figure(figsize=(50,15))
df.boxplot()
plt.show()


# In[73]:


df.columns


# In[ ]:





# In[ ]:





# In[ ]:




