# Basic Stock Market Prediction Models

## 1. Introduction

### 1.1. Background

The stock market is a complex and dynamic system that has fascinated investors and analysts for decades. It is a platform for companies to raise capital and for investors to make profits by buying and selling shares. However, predicting the movement of the stock market is not an easy task, as it is influenced by a multitude of factors such as global events, economic indicators, and company performance. As a result, investors and traders are constantly seeking ways to gain an edge and make informed decisions about their investments. One way to achieve this is by building stock market prediction models, which use historical data and advanced analytical techniques to forecast future market trends.

There are a variety of methods and models to be used as stock market prediction models that are used widely among many projects. In this project, I will investigate the effectiveness of some most common models, thereby providing more insights into their results.

The exemplary data is gathered from Yahoo Finance, and that is the stock price for Alphabet Inc., the parent company of Google. Each row of the dataset represents a single day of trading, with columns containing information about the opening, highest, lowest, and closing prices for that day, as well as the volume of shares traded.

### 1.2. Objectives

* Objective: find out the effectiveness of some most commonly used models for stock market prediction
* Disclaimer: This project is intended to test different models and is not meant to serve as a guide for trading on the stock market.

### 1.3. Tool used

Python:

* Data preparation
* Multiple machine learning models

### 1.4. Data

For this project, my intention is to utilise the most recent data available at the time of doing this project, so I'm extracting the live data from Yahoo Finance, up until June 1, 2023. Otherwise, if you wish to replicate my result, I would suggest that the data be imported via the code below instead of using "data.history(period= 'max')" like I did:
``` python
import yfinance as yf
import pandas as pd

data = yf.Ticker('GOOG')
data = data.history(start= '1900-01-01', end= '2023-06-02')
data.tail(5)
```
Result:

| Date                      | Open      | High      | Low       | Close     | Volume     | Dividends | Stock Splits |
|---------------------------|-----------|-----------|-----------|-----------|------------|-----------|--------------|
| 2004-08-19 00:00:00-04:00 | 2.490664  | 2.591785  | 2.390042  | 2.499133  | 897427216  | 0.0       | 0.0          |
| 2004-08-20 00:00:00-04:00 | 2.515820  | 2.716817  | 2.503118  | 2.697639  | 458857488  | 0.0       | 0.0          |
| 2004-08-23 00:00:00-04:00 | 2.758411  | 2.826406  | 2.716070  | 2.724787  | 366857939  | 0.0       | 0.0          |
| 2004-08-24 00:00:00-04:00 | 2.770615  | 2.779581  | 2.579581  | 2.611960  | 306396159  | 0.0       | 0.0          |
| 2004-08-25 00:00:00-04:00 | 2.614201  | 2.689918  | 2.587302  | 2.640104  | 184645512  | 0.0       | 0.0          |
| ...                       | ...       | ...       | ...       | ...       | ...        | ...       | ...          |
| 2023-05-25 00:00:00-04:00 | 125.209999| 125.980003| 122.900002| 124.349998| 33812700   | 0.0       | 0.0          |
| 2023-05-26 00:00:00-04:00 | 124.065002| 126.000000| 123.290001| 125.430000| 25154700   | 0.0       | 0.0          |
| 2023-05-30 00:00:00-04:00 | 126.290001| 126.379997| 122.889999| 124.639999| 27230700   | 0.0       | 0.0          |
| 2023-05-31 00:00:00-04:00 | 123.699997| 124.900002| 123.099998| 123.370003| 41548800   | 0.0       | 0.0          |
| 2023-06-01 00:00:00-04:00 | 123.500000| 125.040001| 123.300003| 124.370003| 25009300   | 0.0       | 0.0          |

## 2. Prediction Models

In this project, I’m going to use some most common methods and models to try and predict the future stock price. Some methods include:

* Predict the exact closing price
* Predict the change in closing price
* Predict the trend of next day: up or down

The most common model, regularly seen around the data science projects, for stock market prediction is Long Short-Term Memory (LSTM). Along with Autoregressive Integrated Moving Average (ARIMA) (another common model), it will be used to form time series analysis and make future price predictions. As for the trend prediction, 3 models are deployed: Random Forest, XGBoost and Multilayer Perceptron Classifier (MLP).

Let’s take a look at the closing price:

![Closing Price](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/1.%20Closing%20Price.png)

There are notable different trends based on the plot. From 2004-2015, the stock price of GOOG had a more gradual increase with fewer drastic fluctuations, while in the period from 2015 until the time of this project (May 2023), the stock price has shown more rapid and volatile changes. Additionally, the peak values of the stock price were much higher in the more recent period. Thus, I’ve decided to use the data from 2015 only for the models.

### 2.1. Time Series Analysis Models - ARIMA

Autoregressive Integrated Moving Average (ARIMA) is the fundamental basis of time series analysis. Hence, I will use this model for this task in order to test the performance on stock market prediction.

#### 2.1.1. Predicting Closing Price

This model follows tightly the movement of the price. However, it seems like the prediction traces the real price and pick up the previous price for next price, rather than actually predicting, leading to overfitting. This pattern persists across the whole data, and therefore it would not be useful for real-time prediction.
* RMSE: 2.73
* R2 Score: 0.98

![ARIMA Model for Closing Price](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/2.%20ARIMA%20Model%20for%20Closing%20Price.png)

#### 2.1.2. Predicting Price Change

This model avoids the near absolute pattern of the first model, even though it still tracks the previous record as a way to predict the next one. However, it seems that the model predicting price change is not really sensitive to the input of the actual price change, as the result of the prediction does not fluctuate as much as the original change.
* RMSE: 2.80
* R2 Score: -0.18

![ARIMA Model for Price Change](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/3.%20ARIMA%20Model%20for%20Price%20Change.png)

### 2.2. Time Series Analysis Models - LSTM

Long Short-Term Memory (LSTM) is the most recommended model for this particular task across the Internet (as of mid-2023). Thus, I will try and apply this model to the real data. Again, I will try to predict the closing price, and then the price change of the next day.

#### 2.2.1. Predicting Closing Price

In this model, I added one layer of dropout to avoid overfitting like the ARIMA prediction model of closing price. The result is less overfitting like the previous model, but we can witness that the predicted trend is sometimes the lagged version of the original trend.
* RMSE: 3.06
* R2 Score: 0.97

![LSTM Model for Closing Price](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/4.%20LSTM%20Model%20for%20Closing%20Price.png)

#### 2.2.2. Predicting Price Change

I added no dropout layer, and the result is pretty much incompatible with the original price change, even if the RMSE and R2 Score show improvement.
* RMSE: 2.60
* R2 Score: -0.01

![LSTM Model for Price Change](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/5.%20LSTM%20Model%20for%20Price%20Change.png)

### 2.3. Trend Classification Models - Random Forest, XGBoost, & Multilayer Perceptron Classifier

In these models, instead of predicting actual result, I only want to predict the final trend outcome: up or down, so this is basically binary classification. 3 models are deployed to perform this task.

Moreover, to predict the trend of the next day, apart from the opening, high, low, and closing price of the current day, I also add more information regarding the stock market:
* 14-day RSI
* 20-day, 50-day, and 100-day EMA
* Volume of the last day
* Closing price of the last 29 days (making the total values of closing price 30).

#### 2.3.1. Result

All 3 models mostly predict the price increase, indicating that these models, or this method of classification, are probably not the optimal one.
* Random Forest accuracy: 0.57
* XGBoost accuracy: 0.57
* MLP accuracy: 0.55

![Random Forest Confusion Matrix](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/6.%20Random%20Forest%20Confusion%20Matrix.png)

![XGBoost Confusion Matrix](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/7.%20XGBoost%20Confusion%20Matrix.png)

![MLP Classifier Confusion Matrix](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/8.%20MLP%20Classifier%20Confusion%20Matrix.png)

#### 2.3.2. Further Details

Let's take a look at the true prediction value of the classifier. Here is the graph describing the detailed result. We can see that most of the values are above the cutoff of 0.5, but most of them actually fall in a very small range within 0.5-0.6.

![MLP Classifier Further Analysis](https://github.com/nam-anh-21/Basic-Stock-Market-Predictions/blob/main/Images/9.%20MLP%20Classifier%20Further%20Analysis.png)

## 3. Suggestions

As we can all know, the stock market is very dynamic and it is somewhat hard for us to predict it using our knowledge and skills, letalone using Machine Learning/Deep Leaning Models. After I individually tested each of the most basic methods of stock market predicttion, here are some suggestions for further research.

* Use additional models to understand their effectiveness. ARIMA or LSTM are not the only models that can be used for stock market prediction, and there are many more to discover. Some of these models are Generative Adversarial Networks (GAN) or Transformer.
* Combine models together to find the optimal model. In my project, I only tested the models individually, without considering that the model might perform better when combined with others. In fact, there are many different methods of stock market prediction can be found online that combine models together to build a more complexed model and test the performance.
    * One method of a combination of models is applying multiple Neural Networks together, for instance, the hybrid architecture of Convolution Neural Network (CNN) and LSTM.
    * Another method is rather more practical. The stock market fluctuation depends heavily on real-time news, as it can significantly impact investor sentiment and market dynamics. Therefore, relying solely on the previous data may not be sufficient for the actual prediction. More research is necessary for applying Natural Language Processing (NLP) into stock market prediction algorithms, along with data crawling and data mining techniques. NLP helps extracting relevant information that can - or rather will - affect the stock market as a whole.

## 4. Conclusion

In summary, using these models indivisually may not grant us great result in predicting the stock market. Moreover, although these suggestions are perhaps aligned with common practices, it is essential to acknowledge the complexities and uncertainties in predicting stock market movements accurately. For a more robust prediction model, it is advised that a comprehensive approach that combines multiple factors and analysis techniques should be deployed.
