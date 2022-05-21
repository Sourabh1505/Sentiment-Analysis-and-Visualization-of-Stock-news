#!/usr/bin/env python
# coding: utf-8

# In[1]:


from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ["TTM","INFY","WIT"]


# In[3]:


news_tables={}
for ticker in tickers:
    url=finviz_url + ticker
    req = Request(url = url, headers={"user-agent":"my-app"})
    response = urlopen(req)
    html = BeautifulSoup(response,"html")
    news_table= html.find(id="news-table")
    news_tables[ticker] = news_table


# In[4]:


parsed_data =[]
for ticker,news_table in news_tables.items():
    for row in news_table.findAll("tr"):
        title = row.a.text
        date_data = row.td.text.split(" ")
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker,date,time,title])


# In[5]:


df = pd.DataFrame(parsed_data, columns = ["ticker","date","time","title"])
df.head()


# In[6]:


import nltk
nltk.download()


# In[7]:


vader = SentimentIntensityAnalyzer()


# In[8]:


f= lambda title:vader.polarity_scores(title)["compound"]
df["compound"]= df["title"].apply(f)


# In[9]:


df.head()


# In[25]:


df["date"]=pd.to_datetime(df.date)


# In[26]:


df["month"]=df["date"].dt.month


# In[28]:


df["year"]=df["date"].dt.year


# In[31]:


mean_df = df.groupby(["ticker","date"]).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs("compound",axis="columns").transpose()
mean_df.plot(kind="bar",title="XYZ",ylabel="compound",xlabel="date",figsize=(10,10))
plt.show


# In[32]:


mean_df = df.groupby(["ticker","month"]).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs("compound",axis="columns").transpose()
mean_df.plot(kind="bar",title="XYZ",ylabel="compound",xlabel="month",figsize=(5,5))
plt.show


# In[33]:


mean_df = df.groupby(["ticker","year"]).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs("compound",axis="columns").transpose()
mean_df.plot(kind="bar",title="XYZ",ylabel="compound",xlabel="year",figsize=(5,5))
plt.show


# In[17]:


mean_df


# In[ ]:




