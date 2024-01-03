#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First let's import the packages we will use in this project
# You can do this all now or as you need them
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

df = pd.read_csv('/Users/priya/Downloads/movies.csv')


# In[2]:


# Let's explore the data

df.head()


# In[3]:


# Data Cleaning
# Let's see if there is any missing data

for col in df.columns:
    # Check for non-finite values (NaN or inf) and replace them with 0
    col_data = df[col].replace([np.inf, -np.inf, np.nan], 0)
    
    # Calculate the percentage of missing values
    pct_missing = np.mean(col_data.isnull())
    
    print("{} - {:.2f}%".format(col, pct_missing * 100))


# In[4]:


# Data Types

df.dtypes


# In[5]:


# Change data type pf columns


# Assuming 'budget' is the column you want to convert to int64
df['budget'] = pd.to_numeric(df['budget'], errors='coerce', downcast='integer')
df['gross'] = pd.to_numeric(df['gross'], errors='coerce', downcast='integer')
df['votes'] = pd.to_numeric(df['votes'], errors='coerce', downcast='integer')


# In[6]:


df.head()


# In[7]:


df.columns


# In[ ]:





# In[27]:


# Create a sample DataFrame

# Extract the date and country into two separate columns

df[['date', 'country']] = df['released'].str.extract(r'([A-Za-z]+\s\d{1,2},\s\d{4})\s\((.*?)\)')

#Extract last four digit of the date as "yearcorrect"

df['yearcorrect'] = df['date'].str[-4:]

df.head()


# In[29]:


df.sort_values(by=['gross'],inplace=False, ascending=False).head(10)


# In[11]:


pd.set_option('display.max_rows', None)


# In[12]:


# Drop any duplicates

df_company= df['company'].drop_duplicates().sort_values(ascending= False)

df_company.head(10)


# In[13]:


# Budget high correlation
# Company high correlation


# In[14]:


# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget Vs Gross Earnings')

plt.xlabel('Gross Earning')

plt.ylabel('Budget for film')

plt.show()


# In[15]:


# plot the budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, line_kws={"color":"blue"})


# In[16]:


df.corr(method='pearson') #pearson, kendall, spearman


# In[17]:


# High correlation between budget and gross


# In[18]:


# Correlation Matric using Seaborn

correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title ('Correlation Matric for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[19]:


df.head()


# In[30]:


#Categorization of non-numeric data

df_numerized = df.copy()

for col_name in df_numerized.columns:
    if df_numerized[col_name].dtype == 'object':
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

df_numerized.head(10)


# In[31]:


df.head()


# In[22]:


# Correlation matric using Seaborn for all categorized columns

correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)

plt.title ('Correlation Matric for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[23]:


df_numerized.corr()


# In[32]:


# Unstaking correlation matrics 

correlation_matrix = df_numerized.corr()
corr_pairs = correlation_matrix.unstack()
corr_pairs.head(10)


# In[36]:


# Sorting correlation 

sorted_pairs = corr_pairs.sort_values(ascending=False)

sorted_pairs.head(30)


# In[26]:


# Votes and budget have the highest correlation to gross earnings

# Company has low correlation


# # 
