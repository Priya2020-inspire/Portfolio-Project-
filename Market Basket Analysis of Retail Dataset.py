#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('retail_dataset.csv') 

## generate a list of unique items in the market basket
items = (df['0'].unique())

## creates a table with item names as column names and transaction numbers as row names 
## with 1s and 0s indicating item was purchased in transactions
itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row) 
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)

freq_items = apriori(ohe_df, min_support=0.1, use_colnames=True, verbose=1)

rules = association_rules(freq_items, metric='confidence', min_threshold=0.3)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()

x   = rules['confidence']
y   = rules['lift']
m,b = np.polyfit(x, y, 1)
plt.scatter(x, y, alpha = 0.5) 
plt.plot(x,b+m*x, color='red')
plt.xlabel('confidence')
plt.ylabel('lift')
plt.title('Confidence vs Lift')
plt.show()

m=round(m,3)
b=round(b,3)
print('y=',m,"x +",b)


# In[28]:


m=round(m,3)
b=round(b,3)
print('y=',m,"x +",b)


# In[29]:


X=0.65
m*X+b


# In[30]:


most_freq = freq_items.sort_values(by=['support'],ascending=False)
most_freq


# In[31]:


rules.sort_values(by=['confidence'],ascending=False)

