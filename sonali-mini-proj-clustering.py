#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\prane\\Documents\\Univ_of_Bristol\\mini-project\\fake_transactional_data_24.csv')


# In[ ]:





# In[56]:


# Group transactions by account number and sum the transaction amounts
account_sum = df.groupby('from_totally_fake_account')['monopoly_money_amount'].sum().reset_index()
#print(len(account_sum))  8142

# Extract relevant features (assuming 'transaction_amount' is the column of interest)
features = account_sum[['monopoly_money_amount']]

# Apply KMeans clustering
n_clusters = 3  # You may adjust the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
account_sum['cluster'] = kmeans.fit_predict(features)


# In[15]:


print(account_sum.head())


# In[17]:


# Validate count of cluster , total=8142
print(account_sum.groupby('cluster').size().reset_index(name='count'))


# In[22]:


# Validate if a account number belongs to more than one cluster

# Group by account number and count unique clusters
account_cluster_counts = account_sum.groupby('from_totally_fake_account')['cluster'].nunique().reset_index(name='unique_cluster_count')

# Display accounts with more than one cluster
accounts_with_multiple_clusters = account_cluster_counts[account_cluster_counts['unique_cluster_count'] > 1]
print(accounts_with_multiple_clusters)


# In[23]:


# Check the range of each cluster
cluster_ranges = account_sum.groupby('cluster')['monopoly_money_amount'].agg(['min', 'max']).reset_index()

# Display the range for each cluster
print(cluster_ranges)


# In[26]:


# Create a scatter plot with account numbers as labels
import seaborn as sns

# Create a swarm plot
plt.figure(figsize=(15, 8))
sns.swarmplot(x='cluster', y='monopoly_money_amount', data=account_sum, hue='from_totally_fake_account', palette='viridis', size=8, alpha=0.8)
plt.xlabel('Cluster')
plt.ylabel('Transaction Amount')
plt.title('Swarm Plot of Customer Segmentation')
plt.legend(title='Account Number', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[58]:


#Check daily frequency of transaction for fraud detection

df['not_happened_yet_date'] = pd.to_datetime(df['not_happened_yet_date'])

# Create a new column for the day
df['day'] = df['not_happened_yet_date'].dt.date

# Group by account number and day, then calculate the daily transaction frequency
account_daily_counts = df.groupby(['from_totally_fake_account', 'day']).size().reset_index(name='transaction_count')

# Calculate summary statistics (mean and standard deviation) for daily transaction frequency
summary_stats = account_daily_counts.groupby('from_totally_fake_account')['transaction_count'].agg(['mean', 'std']).reset_index()

# Set a threshold for identifying outliers (e.g., transactions more than 3 standard deviations from the mean)
outlier_threshold = summary_stats.copy()
outlier_threshold['outlier_value'] = outlier_threshold['mean'] + 2 * outlier_threshold['std']

# Identify account numbers with unusually high daily transaction frequencies
potential_fraud_accounts = account_daily_counts.merge(outlier_threshold[['from_totally_fake_account', 'outlier_value']], on='from_totally_fake_account', how='inner')
print("data with daily transaction count and outlier value is as follows:")
print(potential_fraud_accounts)
print("=======================================================================")

# Filter accounts with unusually high daily transaction frequencies based on the updated outlier_threshold
potential_fraud_accounts2 = potential_fraud_accounts[potential_fraud_accounts['transaction_count'] > potential_fraud_accounts['outlier_value']]

# Display the result
print("Potential fraud transactions in a day:")
print(potential_fraud_accounts2)


# In[39]:


account_daily_counts


# In[57]:


outlier_threshold


# In[ ]:





# In[ ]:




