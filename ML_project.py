#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd

# The scrapeconfig.py config file contains all xpaths for the websites that are being scraped 
# Since this scraper is modular, you can edit the scrapeconfig.py file to use this 
# scraper to collect data from ANY news website with a results and article page
# Just make sure you set correct XPaths for the properties you want to collect 
from scrapeconfig import *
import multiprocessing

#regex for some string locations 
import re 
import requests
import json

#Parsing CLI arguments
import argparse

#LXML as main HTML parser. Has nice Xpath selection, works everywhere 
from lxml import html
from lxml import etree

#dateparser to handle different types of date formats on articles 
import dateparser as dateParse 
import dateutil.parser as dparser

import csv 
import time 

#Setting default to UTF8 to deal with pesky ascii errors in python 2.x
import sys
reload(sys)
sys.setdefaultencoding("utf8")

def parsedHTML(url):
	#This function handles the web requests and parses the HTML into an lxml tree 
	#Headers so we don't get 403 forbidden errors 
	headers = {
		'accept-encoding': 'gzip, deflate, br',
		'accept-language': 'en-US,en;q=0.8',
		'upgrade-insecure-requests': '1',
		'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
		'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
		'cache-control': 'max-age=0',
		'authority': 'news.bitcoin.com',
		'cookie': '__cfduid=d784026513c887ec39604c0f35333bb231500736652; PHPSESSID=el5c5j7a26njfvoe2dh6fnrer3; _ga=GA1.2.552908756.1500736659; _gid=GA1.2.2050113212.1500736659',
	}
	page = requests.get(url, headers=headers)
	tree = html.fromstring(page.content)
	return tree

def collectArticles(urls, source, args, filename):
	#Loop over all the URLS that were collected in the parent function 
	for url in urls: 

		tree = parsedHTML(url)

		#Initialize empty text string, add paragraphs when collected 
		articleText = ""

		#The function that is called here is from the scrapeconfig.py file (imported)
		#Have to pass the tree along with the source key, otherwise it cant access xpaths 
		print url
		config = pageConfig(source, tree)

		#If page was not found, continue to next URL 
		if not config:
			continue

		#Based on xpaths defined above, call correct selector for current source
		#Could just pass the config selectors to the array, but for the sake of cleanliness...
		
		articleTitle = config['articleTitle']
		articleText = config['articleText']
		articleAuthor = config['articleAuthor']
		#Storing it as a datetime object 
		articleDate = config['articleDate']

		#Check against the year argument, terminate if it turns out the year for the current 
		#article is < than the year you want to collect from (no point in continuing then)
		#if it does not match, don't write, if it's smaller, terminate
		yearToScrape =int(args.scrapeYear)
		s=type(yearToScrape)
		print articleDate
		print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
		print str(dparser.parse(articleDate, fuzzy=True).month)
		print dparser.parse(articleDate,fuzzy=True)

		print("############################################################")
		if (dparser.parse(articleDate,fuzzy=True).year < yearToScrape) :
			break
		elif (dparser.parse(articleDate,fuzzy=True).year != yearToScrape):
			pass
		else:
			print ("In Else \n")
			
			monthToScrape= int(args.scrapeMonth)
			if (dparser.parse(articleDate,fuzzy=True).month == monthToScrape):
				print("IN MONTH"+ str(args.scrapeMonth))
				csvwriter = csv.writer(open(filename, "a"))
				csvwriter.writerow([articleDate, articleTitle, articleAuthor, url, articleText])
			else:
				break

		


def getArticleURLS(source, args):
	#Create filename where everything is stored eventually. Doing str(int()) so the time is rounded off
	filename = source+'_ARTICLES'+'_YEAR_'+str(args.scrapeYear)+'_MONTH_'+str(args.scrapeMonth)+'_'+str(int(time.time()))+'.csv'
	urls = []
	currentPage = 1
	print currentPage
	hasNextPage = True
	outOfRange = False
	while hasNextPage and not outOfRange:
		print 'setting dict'
		#Parse HTML, invoke config (x)paths 
		tree = parsedHTML(resultsConfig(currentPage)[source]['pageURL'])
		print("#################")
		print tree.xpath(resultsConfig(currentPage)[source]['itemXpath'])

		items = tree.xpath(resultsConfig(currentPage)[source]['itemXpath'])

		print 'looping over items'
		print len(items)
		#For every item on the search results page... 
		for item in items:
			#Here we invoke the correct Xpaths from the config dict above 

			#Not every results page correctly displays datetime in result, so if it's not here
			#do the check when fetching the articles. Else, if its ordered by date just terminate if the current article date is < the year youre scraping
			if resultsConfig(currentPage)[source]['dateOnPage'] and resultsConfig(currentPage)[source]['dateOrdered'] and args.scrapeYear:
				articleDate = dparser.parse(item.xpath(resultsConfig(currentPage)[source]['dateXpath'])[0].get('datetime'), fuzzy=True)
				
				#If we already see that the article date is not from a year we want to collect (eg if from 2014 and 2015 was specified)
				#then we just terminate the while loop. Only works one way, as articles are ordered by date, so can only do if smaller 
				yearToScrape =int(args.scrapeYear)
				if articleDate.year < yearToScrape:
					outOfRange = True 
				#Note that it then just terminates on the next page (since there is no 'break' statement for the while loop)

			articleURL = item.xpath(resultsConfig(currentPage)[source]['urlXpath'])[0].get('href')
			
			#Some websites have relative URL pointers, so prefix the base URL 
			if '://' not in articleURL:
				articleURL = resultsConfig(currentPage)[source]['baseURL']+articleURL

			#Urlfilter hack to prevent video/audio/gadfly pages from being visited (mostly bloomberg)
			#These pages have custom xpath structures, so not even bothering collecting them
			urlFilters = ['/videos/','/audio/','/gadfly/','/features/','/press-releases/']
			#If any of the above strings is in the url, pass writing it, else write it 
			if any(urlFilter in articleURL for urlFilter in urlFilters):
				pass
			else:
				urls.append(articleURL)

		#If there are less items in the results than the resultsPerPage param, we assume this is the last page 
		if len(items) < resultsConfig(currentPage)[source]['resultsPerPage']:
			hasNextPage = False 

		#Increase page number by 1 for the next iteration of the while loop 
		currentPage += 1

		#Once all URLs for the page have been collected, go visit the actual articles 
		#Do this here so it doesn't first collect too many URLs that are useless afterwards 
		collectArticles(urls, source, args, filename)
		#Reinitialize URLS array again for next loop 
		urls = []


if __name__ == '__main__':

	#Neat way of inputting CLI arguments 
	parser = argparse.ArgumentParser(description='Scrape news articles')
	parser.add_argument("--year", dest="scrapeYear", required=False, help="Specify a specific year to collect from")
	parser.add_argument("--month", dest="scrapeMonth", required=False, help="Specify a specific month to collect from")
	parser.add_argument('--sources', nargs='+', dest="sources", help='Set the news websites you want to collect from', required=False)
	args = parser.parse_args()
	print args.scrapeYear
	print args.sources
	print args.scrapeMonth

	#Check if some sources are defined as input argument, otherwise just go over all 
	allSources = ['coindesk','reuters','newsbitcoin','wsj','cnbc','bloomberg']
	if args.sources:
		visitSources = args.sources
	else:
		visitSources = allSources

	for source in visitSources:		
		#Using multiprocessing to speed things up a little. Creates new process thread for every source channel o
		#Calling getArticleURLS will also call child function that collects the actual articles 
		p = multiprocessing.Process(target=getArticleURLS, args=(source, args))
		p.start()
		print 'started thread'

#~~~~~~~~~~~~~~~~~~End of Scraper code~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Bitcoin Prices Historical Data Table Scraping		
bitcoin_price=pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20171017') #scrapping Bitcoin prices from website
for df in bitcoin_price: 
    print(df)
df.head()  

statistics.mean((df['High'])) #checking general stats for bitcoin prices
statistics.stdev((df['High']))
df.to_csv('Bitcoin_historical.csv') #exporting files to csv
print(df.describe())#df is the name of the Bitcoin Historical Prices Data Frame


#combining all csv from the scraper to the list
import glob
coindesk_files = glob.glob('/Users/vaghanideep/Downloads/News/coindesk/*.csv')
df_list = []
for filename in sorted(interesting_files):
    df_list.append(pd.read_csv(filename))
full_df = pd.concat(df_list)
full_df.to_csv('output.csv')
#~~~~~~~~~~~~End of Historical prices Data information~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Pulling one source of news information from Coindesk.com
#All news from coindesk
data =pd.read_csv("/Users/vaghanideep/Downloads/News/coindesk/out.csv", 
                    names=["Timestamp","Heading","Author","Link","Contents"])
                    
#~~~~~~~~~~~General data information and learning features and variables (Coindesk.com)~~~~~~~~~~~~~~~~~~                    

#Data Cleaning Tasks for Coindesk news
data2=data.copy()
data2.describe() #general stats of data
data2.dtypes #types of data in dataframe
data2.T #transpose the data

str(data2.Timestamp)
type(data2['Timestamp'][0])
data2.Timestamp.apply(str)
data2['Year'] = pd.DatetimeIndex(data2['Timestamp']).year
data2['Date'] = pd.DatetimeIndex(data2['Timestamp']).date
data2[['Date','Time']] = data2.Timestamp.str.split(expand=True)
data2['Timestamp'].head()
data2['Timestamp'].dtypes
data2['Timestamp'].value_counts()  #Getting Frequency of Timestamps
newsbitcoin['Author'].value_counts()  #Getting Frequency of Athors
data2['Author'].value_counts()

#counting unique authors freqency
my_tab = pd.crosstab(index=data2["Author"],        # Make a crosstab to get idea on frequency of authors and publications
                              columns="count") 
                              
                              

#pivoting dataframe with required frequency counts
temp_df_2 = data2.groupby(by = ['Date', 'Author']).size().unstack()
ml_coindesk=temp_df_2.copy() #copy of temp_df and using it for model  

ml_coindesk=ml_coindesk.fillna(0) #filling Nan values with '0'
ml_coindesk['No of Publications']= ml_coindesk.sum(axis=1) #Count of number of publications
np.isfinite(ml_coindesk.columns)# Checking elementwise finiteness of the data
df['Date'].dtypes
ml_coindesk['Date'].dtypes
ml_coindesk = ml_coindesk.reset_index()# converting index into column in pandas


#~~~~~~~~Merging and creating the ML data set for running different ML algorithms~~~~~~~~~
df.Date = pd.to_datetime(df.Date)
ml_coindesk.Date=pd.to_datetime(ml_coindesk.Date)

result_1= pd.merge(ml_coindesk,df,on=['Date','Date']) 
ml_coindesk['Date'] = ml_coindesk.index

result_2=result_1.copy() #result_2 is the first dataset from which training datasets will be created

result_2['Diff']=result_2['Close']-result_1['Open'] #Diff column is created

result_2['Per_chg']=(result_2['Diff']*100/(result_2['Open'])) #Per_chg column is created to check what percentage of Open is difference value.

result_2['Per_chg'].describe()

del result_2['Param'] 

result_2['Param']=[0]*result_2.shape[0] #Param column where three parameters up,down,same are created.

for i,j in enumerate(result_2['Per_chg']):
    if j > 10:
        result_2.loc[i, 'Param']=2
    elif j < 0:
        result_2.loc[i, 'Param']=1
    else:
        result_2.loc[i, 'Param']=0
        
#~~~~~~~Running Linear Regression

#############################ML################################################
#doing linear regression on The new data frame 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X=result_2
X['Index1'] = X.index

del X['Diff']
del X['Open']
del X['High']
del X['Low']
del X['Close']
del X['Market Cap']
del X['Volume']
del X['Date']
del X[' Diff1%']
del X['Diff 1%']
#Deleting unwanted features from our X training sets

str(X.Date)


Y=result_2['Param'] #Target variables which is Param

lm=LinearRegression()
lm.fit(X,Y)

print('Estimated intercept coefficient',lm.intercept_)
print('Number of coefficeints',len(lm.coef_))

print(Y,lm.coef_)

type(df)
type(df['High'])


plt.scatter(X.High,df_price)
lm.predict(X)[1:15]

plt.scatter(df_price,lm.predict(X))
mseFull=np.mean((df_price - lm.predict(X))**2)
print(mseFull)


#splitting data in training models for better capturization of mean square error
X_train,X_test,Y_train,Y_test =sklearn.cross_validation.train_test_split(X, Y, test_size=0.33,random_state=5)
print(Y_train.shape)


#predicitng trained models
lm.fit(X_train,Y_train)
pred_train=lm.predict(X_train)
pred_test=lm.predict(X_test)
print(pred_train)


#checking mean squared error
np.mean((Y_train - lm.predict(X_train))** 2)
np.mean((Y_test - lm.predict(X_test))** 2)


#Plotting residual plots
plt.scatter(lm.predict(X_train),lm.predict(X_train) - Y_train, c='b', s=40,alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y=0,xmin=0,xmax=50)
plt.title('Residual plot using training (blue) and test (green) data')
plt.ylabel('Residuals')


#calcuating r2_score to check the model performance
r2_score(Y_train, pred_train)
lm.get_params([lm.coef_])        
        
#Achieving 0.75 r2 score which is promising and significant.


#~~~~~~~~~Preforming Naive Bayes ML techniques
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

Y=result_2['Param']

X=result_2
X['Index1'] = X.index

print(clf.fit(X, Y))

X_train,X_test,Y_train,Y_test =sklearn.cross_validation.train_test_split(X, Y, test_size=0.20,random_state=5)

clf.fit(X, Y)
clf.score(X, Y)

clf.fit(X_train,Y_train)

clf.predict(X_test)

plt.scatter(clf.predict(X_train),clf.predict(X_train) - Y_train, c='b', s=40,alpha=0.5)

clf.score(X_test, Y_test)

#Achieving 0.59 r2 which is promising and iterating further to better the accuracy score.





#~~~~~~~~~~~~~~~ Bringing in another source (News.bitcoin.com) ~~~~~~~~~~~~~~
    
newsbitcoin=pd.read_csv("/Users/vaghanideep/Downloads/News/bloomberg/merged.csv",
                         names=["Timestamp","Heading","Author","Link","Contents"])
#cleaning data

newsbitcoin.dtypes

newsbitcoin.Timestamp.value_counts().sort_index()#sort patterns in one place

newsbitcoin.columns#getting columns name

newsbitcoin.rename(columns=lambda x: x.strip(), inplace=True) #striping whitespaces in column for ease of use

newsbitcoin['xx'] = newsbitcoin['xx'].str.replace(',', '') #to remove any commas in the columns

type(newsbitcoin['Timestamp'][0])#confirming to check timestamp is str type

newsbitcoin['Date'] = pd.DatetimeIndex(newsbitcoin['Timestamp']).date #extracting date from timestamp

newsbitcoin=newsbitcoin.dropna()#removing rows containign atleast one nan value which is useless for current analysis

newsbitcoin.Author= newsbitcoin.Author.fillna('Unknown')#filling nan values with unknown

##now making data as per required format using unstack()
newsbitcoin['Author'].value_counts

newsbitcoin.Author.unique

temp_df_3 = newsbitcoin.groupby(by = ['Date', 'Author']).size().unstack()

newsbitcoin.groupby('Author')['Date'].nunique()

temp_df_3= temp_df_3.reset_index()# converting index into column in pandas

temp_df_3=temp_df_3.fillna(0)

temp_df_3['No of Publications']= temp_df_3.sum(axis=1) #number of publications


##~~~~~~~~~~~~~~~~Copy of news bitcoin paramater~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
temp_df_copy=temp_df_3.copy()

temp_df_copy.info()
temp_df_copy.columns

temp_df_copy.rename(columns=lambda x: x.strip(), inplace=True) #striping whitespaces in column for ease of use
temp_df_copy['Adam Satariano'], temp_df_copy['Nate Lanxon'] = temp_df_copy['Adam Satariano & Nate Lanxon'].str.split(' ', 1).str
temp_df_copy['x_new'] = temp_df_copy['x'].str.split('-')

import numpy as np
import pandas as pd

col_copy = temp_df_copy.filter(regex='&')
temp_df_copy1= temp_df_copy[temp_df_copy.columns.drop(list(temp_df_copy .filter(regex='&')))]
del temp_df_copy1['No of Publications']
temp_df_copy1['No of Publications']= temp_df_copy1.sum(axis=1) #number of publications

#Merging Data from historical prices and newsbitcoin

df.Date = pd.to_datetime(df.Date)
temp_df_copy1.Date=pd.to_datetime(temp_df_copy1.Date)

#ML data test model
nb_result_1= pd.merge(temp_df_copy1,df,on=['Date','Date'])

nb_result_1['Diff']=nb_result_1['Close']-nb_result_1['Open']

nb_result_1['Per_chg']=(nb_result_1['Diff']*100/(nb_result_1['Open']))

nb_result_2=nb_result_1.copy()

nb_result_1['Param']=[0]*nb_result_2.shape[0]
del nb_result_2['Param']

for i,j in enumerate(nb_result_1['Per_chg']):
    if j > 10:
        nb_result_1.loc[i, 'Param']=1
    elif j < 0:
        nb_result_1.loc[i, 'Param']=-1
    else:
        nb_result_1.loc[i, 'Param']=0


del nb_result_2['Open']
del nb_result_2['Close']
del nb_result_2['High']
del nb_result_2['Low']
del nb_result_2['Market Cap']
del nb_result_2['Volume']
del nb_result_2['Diff']

#~~~~~Naive Bayes nb ~~~~~~~
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

Y_nb=nb_result_2['Param']

X_nb=nb_result_2

del X_nb['Param']
del X_nb['Per_chg']
del X_nb['Date']
X['Index1'] = X.index

clf.fit(X_nb, Y_nb)
print(clf.fit(X_nb, Y_nb))

X_train,X_test,Y_train,Y_test =sklearn.cross_validation.train_test_split(X_nb, Y_nb, test_size=0.20,random_state=5)

clf.fit(X_train,Y_train)


clf.score(X, Y)

clf.predict(X_test)

plt.scatter(clf.predict(X_train),clf.predict(X_train) - Y_train, c='b', s=40,alpha=0.5)

clf.score(X_test, Y_test)
#0.58333333333333337
  
#After performing the same test with different source and achieving same results we are going after something and 
#keep iterating till we have significant results and improving features and target variables to continuing working on research paper.                   
                              

