#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:41:26 2017

@author: vaghanideep
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
========================================================="""

#importing necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#getting historical price data
bitcoin_price=pd.read_html('https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20171023') #scrapping Bitcoin prices from website
for df in bitcoin_price: 
    print(df)
df.head()  
statistics.mean((df['High'])) #checking general stats for bitcoin prices
statistics.stdev((df['High']))

#converting bunch of rows to float from series and string.
pd.to_numeric(df['Open'], downcast='float')
pd.to_numeric(df['Close'], downcast='float')
pd.to_numeric(df['High'], downcast='float')
pd.to_numeric(df['Low'], downcast='float')
pd.to_numeric(df['Market Cap'], downcast='float')
pd.to_numeric(df['Volume'], downcast='float')
pd.to_numeric(df['date_delta'], downcast='float')


pd.to_numeric(df_price, downcast='float')


df_price=df['Open']
df_date=df['date_delta']


X= df.drop('Open',axis=1)
X=X.drop('Volume', axis=1)


lm=LinearRegression()
lm.fit(X,df_price)

print('Estimated intercept coefficient',lm.intercept_)
print('Number of coefficeints',len(lm.coef_))

data_x=zip(X.columns,lm.coef_)
pd.DataFrame(data_x)

print(data_x)
lm.coef_


type(df)
type(df['High'])


plt.scatter(X.High,df_price)
lm.predict(X)[1:5]

plt.scatter(df_price,lm.predict(X))
mseFull=np.mean((df_price - lm.predict(X))**2)
print(mseFull)


#splitting data in training models for better capturization of mean square error
X_train,X_test,Y_train,Y_test =sklearn.cross_validation.train_test_split(X, df_price, test_size=0.33,random_state=5)
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


