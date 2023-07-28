#!/usr/bin/env python
# coding: utf-8

# # Market Basket Analyis in Python using Apriori Algorithm

# ![image](https://github.com/2110030351/CodeClause_Market_Basket_Analyis_in_Python_using_Apriori_Algorithm/assets/109647212/ff5cf1a4-0a5a-48f9-8e66-cde91fdb9528)
# 
# **Problem Statement:**
# Market Basket Analysis is a powerful modeling technique used in the telecom sector to understand customer purchasing behaviors. It is based on the theory that certain groups of items are more likely to be bought together. For example, customers who buy a pint of beer without a bar meal are more likely to purchase crisps. By analyzing customer itemsets, we can identify relationships between purchases and predict what customers might buy next based on their current purchases.
# The objective of this project is to perform Market Basket Analysis using the Apriori Algorithm on telecom customer transaction data. By discovering associations between different products, telecom providers can strategically improve customer retention and optimize sales.
# 
# **Methodology:**
# _**1. Data Collection:**_
#    - Gather transactional data containing customer purchases, including product codes and timestamps.
# 
# _**2. Data Preprocessing:**_
#    - Cleanse and preprocess the data to remove duplicates and handle missing values.
# 
# _**3. Market Basket Analysis:**_
#    - Implement the Apriori Algorithm to identify frequent itemsets and generate association rules.
# 
# _**4. Rule Evaluation:**_
#    - Calculate the support and confidence of association rules to assess their significance.
# 
# _**5. Location and Promotion Optimization:**_
#    - Use the analysis results to decide the location and promotion of products inside telecom stores.
#    - Place high-margin products near frequently purchased items to increase customer temptations.
# 
# _**6. Differential Analysis:**_
#    - Compare results between different stores, customer demographic groups, and time periods to identify unique patterns.
#    - Investigate differences to gain insights into customer preferences and optimize sales strategies.
# 
# _**7. Visualization and Reporting:**_
#    - Present the findings using charts, graphs, and reports for better comprehension and decision-making.

# In[8]:


get_ipython().system('pip install squarify')


# In[19]:


#pip install mlxtend
pip install wordcloud


# In[23]:


pip install --upgrade wordcloud Pillow


# In[29]:


pip show wordcloud Pillow


# # Importing libraries

# In[51]:


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
plt.style.use('fivethirtyeight')
import os

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

data = pd.read_csv(r'C:\Users\manas\Downloads\Market_Basket_Optimisation.csv',header = None)
data.head()


# In[14]:


data.shape


# In[15]:


data.tail()


# In[16]:


# checking the random entries in the data
data.sample(10)


# In[17]:


data.describe()


# # Data Visualization

# In[32]:


import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
from wordcloud import WordCloud
wordcloud = WordCloud().generate('Your text here')

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data[0]))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Items',fontsize = 20)
plt.show()


# In[35]:


y = data[0].value_counts().head(50).to_frame()
y.index


# In[33]:


plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
data[0].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()


# In[36]:


plt.rcParams['figure.figsize'] = (20, 20)
color = plt.cm.cool(np.linspace(0, 1, 50))
squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)
plt.title('Tree Map for Popular Items')
plt.axis('off')
plt.show()


# In[38]:


data['food'] = 'Food'
food = data.truncate(before = -1, after = 15)
food = nx.from_pandas_edgelist(food, source = 'food', target = 0, edge_attr = True)
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(food)
color = plt.cm.Wistia(np.linspace(0, 15, 1))
nx.draw_networkx_nodes(food, pos, node_size = 15000, node_color = color)
nx.draw_networkx_edges(food, pos, width = 3, alpha = 0.6, edge_color = 'black')
nx.draw_networkx_labels(food, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 15 First Choices', fontsize = 40)
plt.show()


# In[41]:


data['secondchoice'] = 'Second Choice'
secondchoice = data.truncate(before = -1, after = 15)
secondchoice = nx.from_pandas_edgelist(secondchoice, source = 'food', target = 1, edge_attr = True)


plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(secondchoice)
color = plt.cm.Blues(np.linspace(0, 15, 1))
nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = 'pink')
nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.6, edge_color = 'black')
nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 15 Second Choices', fontsize = 40)
plt.show()


# In[50]:


data['thirdchoice'] = 'Third Choice'
secondchoice = data.truncate(before = -1, after = 10)
secondchoice = nx.from_pandas_edgelist(secondchoice, source = 'food', target = 2, edge_attr = True)


plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(secondchoice)
color = plt.cm.Reds(np.linspace(0, 15, 1))
nx.draw_networkx_nodes(secondchoice, pos, node_size = 15000, node_color = color)
nx.draw_networkx_edges(secondchoice, pos, width = 3, alpha = 0.8, edge_color = 'brown')
nx.draw_networkx_labels(secondchoice, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 10 Third Choices', fontsize = 40)
plt.show()


# # Data Preprocessing

# In[53]:


# making each customers shopping items an identical list
trans = []
for i in range(0, 7501):
    trans.append([str(data.values[i,j]) for j in range(0, 20)])
trans = np.array(trans)
print(trans.shape)


# # Using Transaction encoder

# In[54]:


te = TransactionEncoder()
data = te.fit_transform(trans)
data = pd.DataFrame(data, columns = te.columns_)
data.shape


# In[55]:


data = data.loc[:, ['mineral water', 'burgers', 'turkey', 'chocolate', 'frozen vegetables', 'spaghetti',
                    'shrimp', 'grated cheese', 'eggs', 'cookies', 'french fries', 'herb & pepper', 'ground beef',
                    'tomatoes', 'milk', 'escalope', 'fresh tuna', 'red wine', 'ham', 'cake', 'green tea',
                    'whole wheat pasta', 'pancakes', 'soup', 'muffins', 'energy bar', 'olive oil', 'champagne', 
                    'avocado', 'pepper', 'butter', 'parmesan cheese', 'whole wheat rice', 'low fat yogurt', 
                    'chicken', 'vegetables mix', 'pickles', 'meatballs', 'frozen smoothie', 'yogurt cake']]
data.shape


# In[56]:


data.columns


# In[57]:


data.head()


# # Applying apriori algorithum

# Step 1: Create a frequency table of all the items that occur in all the transactions.
# 
# Step 2: We know that only those elements are significant for which the support is greater than or equal to the threshold support.
# 
# Step 3: The next step is to make all the possible pairs of the significant items keeping in mind that the order doesn’t matter, i.e., AB is same as BA.
# 
# Step 4: We will now count the occurrences of each pair in all the transactions.
# 
# Step 5: Again only those itemsets are significant which cross the support threshold
# 
# Step 6: Now let’s say we would like to look for a set of three items that are purchased together. We will use the itemsets found in step 5 and create a set of 3 items.
# 
# ![image.png](attachment:image.png)

# In[58]:


apriori(data, min_support = 0.01, use_colnames = True)


# In[59]:


#For filtering and selecting results
frequent_itemsets = apriori(data, min_support = 0.05, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets


# In[60]:


frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.01) ]


# In[61]:


frequent_itemsets[ (frequent_itemsets['length'] == 1) & (frequent_itemsets['support'] >= 0.01) ]


# # Association Mining

# In[62]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'eggs', 'mineral water'} ]


# In[63]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'mineral water'} ]


# In[64]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'milk'} ]


# In[65]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'chicken'} ]


# In[66]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'frozen vegetables'} ]


# In[67]:


frequent_itemsets[ frequent_itemsets['itemsets'] == {'chocolate'} ]

