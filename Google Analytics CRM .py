#!/usr/bin/env python
# coding: utf-8

# # Install Packages

# In[46]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as datetime
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import gc
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from flask import request
from datetime import datetime


# # Run data

# In[10]:


json_data=["device","geoNetwork","totals","trafficSource"]

gc.enable()

features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',       'visitNumber', 'visitStartTime', 'device_browser',       'device_deviceCategory', 'device_isMobile', 'device_operatingSystem',       'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',       'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region',       'geoNetwork_subContinent', 'totals_bounces', 'totals_hits',       'totals_newVisits', 'totals_pageviews', 'totals_transactionRevenue',       'trafficSource_adContent', 'trafficSource_campaign',       'trafficSource_isTrueDirect', 'trafficSource_keyword',       'trafficSource_medium', 'trafficSource_referralPath',       'trafficSource_source','customDimensions']

def load_df(csv_path):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                    chunksize = 100000)
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        use_df = df[features]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print(ans.shape)
    return ans


# In[11]:


train = load_df('Desktop/train_v2.csv')


# In[12]:


train.head()


# # Deal with missing value 

# In[13]:


def missing_values(data):
    total = data.isnull().sum().sort_values(ascending = False) # getting the sum of null values and ordering
    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) #getting the percent and order of null
    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Concatenating the total and percent
    print("Total columns at least one Values: ")
    print (df[df['Total'] != 0]) # Returning values of nulls different of 0
    return 
missing_values(train) 


# In[34]:


#Deal with missing data 

#fill in NULL transaction revenue with zero 
train["totals_pageviews"]= train["totals_pageviews"].fillna(1,inplace=True).astype(int)
# Replace NA new visits with 0
train["totals_newVisits"]=train["totals_newVisits"].fillna(0,inplace=True)
#Replace NA bounce with 0
train["totals_bounces"]=train["totals_bounces"].fillna(0, inplace=True)
#Revenue Unknown replace with 0
train["totals_transactionRevenue"] = train["totals_transactionRevenue"].fillna(0.0).astype(float)
# fillna object feature
for col in ['trafficSource_keyword',
            'trafficSource_referralPath',
            'trafficSource_adContent']:
    train[col].fillna('unknown', inplace=True)

# fillna boolean feature
train['trafficSource_isTrueDirect'].fillna(False, inplace=True)


# # drop constant column

# In[15]:


constant_column = [col for col in train.columns if train[col].nunique() == 1]
for c in constant_column:
    print(c + ':', train[c].unique())

print('drop columns:', constant_column)
train.drop(constant_column, axis=1, inplace=True)


# # Custom Dimensions

# In[19]:


train['customDimensions'].unique()


# # Data Plot

# ### Device VS transactions

# In[20]:


def chats(data):
    trace = go.Bar(y=data.index[::-1],
                   x=data.values[::-1],
                   showlegend=False,
                   orientation = 'h',
    )
    return trace

data=train.groupby("device_browser")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace1=chats(data["count"].head(10))
trace2=chats(data["count of non-zero revenue"].head(10))
trace3=chats(data["mean"].head(10))


data=train.groupby("device_deviceCategory")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace4=chats(data["count"].head(10))
trace5=chats(data["count of non-zero revenue"].head(10))
trace6=chats(data["mean"].head(10))


data=train.groupby("device_operatingSystem")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace7=chats(data["count"].head(10))
trace8=chats(data["count of non-zero revenue"].head(10))
trace9=chats(data["mean"].head(10))


# In[21]:


fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 
                          subplot_titles=["device_browser - Count", "device_browser - Non-zero Revenue Count", "device_browser - Mean Revenue",
                                          "Device Category - Count",  "Device Category - Non-zero Revenue Count", "Device Category - Mean Revenue", 
                                          "Device OS - Count", "Device OS - Non-zero Revenue Count", "Device OS - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
plotly.offline.plot(fig, filename='device-plots.html')


# ### Conclusion 
# <font color= darkblue>
# 1. Chrome provides the highest non-zero recenue, however customers who uses Firefox has the highest mean revenue
#     
# 2. Customers who uses desktop will more willing to purchase more products in google shopping
# 
# 3. Window user tends to purchase but chrome OS user will buy more items than windows users

# ### Geographic Info VS Transactions

# In[23]:


def chats(data):
    trace = go.Bar(y=data.index[::-1],
                   x=data.values[::-1],
                   showlegend=False,
                   orientation = 'h',
    )
    return trace
data=train.groupby("geoNetwork_city")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace1=chats(data["count"].head(10))
trace2=chats(data["count of non-zero revenue"].head(10))
trace3=chats(data["mean"].head(10))


data=train.groupby("geoNetwork_continent")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace4=chats(data["count"].head(10))
trace5=chats(data["count of non-zero revenue"].head(10))
trace6=chats(data["mean"].head(10))


data=train.groupby("geoNetwork_country")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
data=data.sort_values(by="count",ascending=False)
trace7=chats(data["count"].head(10))
trace8=chats(data["count of non-zero revenue"].head(10))
trace9=chats(data["mean"].head(10))


# Creating two subplots
fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 
                          subplot_titles=["geoNetwork_city - Count", "geoNetwork_city - Non-zero Revenue Count", "geoNetwork_city- Mean Revenue",
                                          "geoNetwork_continent - Count",  "geoNetwork_continent - Non-zero Revenue Count", "geoNetwork_continent - Mean Revenue", 
                                          "geoNetwork_country - Count", "geoNetwork_country - Non-zero Revenue Count", "geoNetwork_country - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)
fig.append_trace(trace7, 3, 1)
fig.append_trace(trace8, 3, 2)
fig.append_trace(trace9, 3, 3)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Geographical Plots")
plotly.offline.plot(fig, filename='Geographical.html')


# 
# ### Conclusion 
# <font color=darkblue>
# 1. Customers in NEW YORK AND SF are the main revenue stream 
#   
# 2. US generates the most revenue but Asia has relative large purchase population
# 
# 3. Canada also has relative high revenue

# In[29]:


def chats(data):
    trace = go.Scatter(x=data.index[::-1],
                   y=data.values[::-1],
                   mode='markers'
    )
    return trace


data=train.groupby("totals_hits")["totals_transactionRevenue"].agg(["size","count","mean"])
data.columns=["count", "count of non-zero revenue", "mean"]
trace1=chats(data["count"])
trace2=chats(data["count of non-zero revenue"])
trace3=chats(data["mean"])


# In[36]:



data=train.groupby("totals_pageviews")["totals_transactionRevenue"].agg(["count","mean"])
data.columns=[ "count of non-zero revenue", "mean"]
trace5=chats(data["count of non-zero revenue"])
trace6=chats(data["mean"])
# Creating two subplots
fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.04, 
                          subplot_titles=[
                                            "totals_hits - Non-zero Revenue Count", "totals_hits - Mean Revenue", 
                                          "totals_pageviews - Count", "totals_pageviews - Non-zero Revenue  Count", "totals_pageviews - Mean Revenue"])

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 1, 3)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 2)
fig.append_trace(trace6, 2, 3)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Webpage Performance Plots")
plotly.offline.plot(fig, filename='Webpage Performance.html')


# In[ ]:





# ### Time Series Revenue Performance

# In[50]:


format_str = '%Y%m%d'
train['formated_date'] = train['date'].apply(lambda x: datetime.strptime(str(x), format_str))
train['_year'] = train['formated_date'].apply(lambda x:x.year)
train['_month'] = train['formated_date'].apply(lambda x:x.month)
train['_quarterMonth'] = train['formated_date'].apply(lambda x:x.day//8)
train['_day'] = train['formated_date'].apply(lambda x:x.day)
train['_weekday'] = train['formated_date'].apply(lambda x:x.weekday())


# In[84]:


data=train.groupby("_weekday")["totals_transactionRevenue"].agg("mean")
data.to_frame()


# In[90]:


def chats(data):
    trace = go.Scatter(x=data.index[::-1],
                   y=data.values[::-1],
                   mode='markers'
    )
    return trace


# In[96]:


data=train.groupby("_month")["totals_transactionRevenue"].agg("mean")
data.columns=["mean"]
trace3=chats(data)


# In[97]:


trace3


# In[85]:





# In[ ]:




