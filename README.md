# Machine-Learning-for-Cryptocurrencies
As a part of my Independent study, I worked on performing Machine Learning on the cryptocurrencies. 
My project was divided into two parts. 

1- I performed linear classifier on the historical cryptocurriencies prices and made prediction function using the classifier. This was used as the motivation for my next steps as linear model only worked till January 2017, after that the price increase was just too volatile for model to handle and I felt need to bring in dfferent factors to make it more accurate.

2- I performed trend analysis and news text mining to create function to predict direction of Bitcoin prices based on news, 
google search trends and unique authors from news articles. This made me realize, only considering social sentiments will not be good as professional traders tend to make decisions based on well written technical articles by regular authors. I scraped all news articels from Jan 2017 to October 2017 from news sources (bloomberg.com, coindesk.com, news.bitcoin.com) and included frequency of articles published on daily basis with prices and trained model to predict the direction of prices by factor of 0.5% to 1% from previous close.

Performed C.V and created co-relation matrix and achieved r-squared of 0.85. 
