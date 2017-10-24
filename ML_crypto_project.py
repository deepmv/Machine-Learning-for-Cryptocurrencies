#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:13:33 2017

@author: vaghanideep
"""

from bs4 import BeautifulSoup #importing Beautifulsoup and other libraries
import requests
import statistics
import pandas as pd
r  = requests.get("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20171023")
r.content
soup = BeautifulSoup(r.content, 'html.parser')
print(soup.prettify())

bitcoin_price=pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20171023') #scrapping Bitcoin prices from website
for df in bitcoin_price: 
    print(df)
df.head()  
statistics.mean((df['High'])) #checking general stats for bitcoin prices
statistics.stdev((df['High']))
df.to_csv('Bitcoin_historical.csv') #exporting files to csv

print(df.describe())

# =============================================================================
# soup.find_all('p')
# soup.find_all('p')[0].get_text()
# historical = soup.find(id="historical-data")
# table_price = historical.find_all(class_="tab-pane active")
# period = table_price.find(class_="period-name").get_text()
# 
# table=soup.find_all('table')[0]
# print(table)
# table_rows=table.find_all('tr')
# for tr in table_rows:
#     td=tr.find_all('td')
#     row=[i.text for i in td]
#     print(row)
# 
# =============================================================================


import os
import glob


def concatenate(indir="/Users/vaghanideep/Downloads/News/coindesk", outfile="/Users/vaghanideep/Downloads/News/out/coindeskcombined.csv"):
    os.chdir(indir)
    filesList=glob.glob("*.csv")
    dfList=[]
    colnames=["Timestamp","Heading","Author","Link","Contents"]
    for filename in filesList:
        print(filename)
        df=pd.read_csv(filename,Header=None)
        dfList.append(df)
    concatDf=pd.concat(dfList,axis=0)
    concatDf.columns=colnames
    concatDf.to_csv(outfile,index=None)



import numpy as np
import pandas as pd
from nltk import word_tokenize 


import glob
coindesk_files = glob.glob('/Users/vaghanideep/Downloads/News/coindesk/*.csv')
df_list = []
for filename in sorted(interesting_files):
    df_list.append(pd.read_csv(filename))
full_df = pd.concat(df_list)

full_df.to_csv('output.csv')



#Import CSV files 
getnews=pd.read_csv("/Users/vaghanideep/Downloads/News/coindesk/coindesk_ARTICLES_YEAR_2017_MONTH_2_1508291436.csv", 
                    names=["Timestamp","Heading","Author","Link","Contents"])

content=(getnews['Contents'])


import re
import string
frequency = {}

document_text =crypto2txt
text_string = document_text.lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1
     
frequency_list = frequency.keys()
 
for words in frequency_list:
    print (words, frequency[words])

print(frequency['bitcoin'])
print(frequency['blockchain'])
print(frequency['ethereum'])
print(frequency['cryptocurrency'])
print(frequency['altcoins'])










