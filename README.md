# A Regression Model for FreqAI
A [regression model](https://github.com/Netanelshoshan/freqAI-LSTM)
for  [FreqAI](https://www.freqtrade.io/en/stable/freqai/) module
from [freqtrade](https://github.com/freqtrade/freqtrade), a crypto trading platform.

Designed to predict outcomes using the LSTM neural network architecture, combined with attention mechanisms.

## Dependencies

Make sure to have `keras-tuner`,`tensorflow` and `tensorflow-addons` installed.
if you don't have them installed, you can install tf using
the [following guide](https://www.tensorflow.org/install/pip).

#### Apple silicon support requires to install tensorflow-metal

## Model Architecture

The model is built using Keras:

1. **Input Layer**: Accepts a 3D tensor representing samples, timestamps, and features.
2. **LSTM Layers with Attention Mechanism**: Multiple LSTM layers can be stacked based on the configuration. Each LSTM
   layer is followed by an attention mechanism which weighs the importance of each timestamp in the sequence.
3. **TimeDistributed Layer**: A fully connected layer to process the output of the LSTM layers.
4. **Dropout**: Helps in preventing overfitting.
5. **Flatten Layer**: Flattens the 3D tensor to 2D.
6. **Dense Layer**: Another fully connected layer.
7. **Residual Connection**: Connects the output of the dense layer to the input to allow for skip connections.
8. **Output Layer**: Produces the final predictions.

# Strategy Formula

The strategy is built upon various technical indicators to determine the most optimal trading decisions. The core of the
strategy is a scoring
system, which combines these indicators to produce a single score, denoted by **`T`**, for each time point in the data.

## Technical Indicators

The strategy employs the following technical indicators with their respective time periods:

- **Simple Moving Average (SMA)**: with a time period of 10. It is the unweighted mean of the previous `n` data points.
  
![SMA](https://latex.codecogs.com/svg.image?{SMA}(t,n)=\frac{1}{n}\sum_{i=0}^{n-1}\text{close}(t-i))


-**Rate of Change (ROC)**: With a time period of 2. It measures the percentage change in price between the current
  price and the price `n` periods ago.
  

![ROC](https://latex.codecogs.com/svg.image?{ROC}(t,n)=\frac{\text{close}(t)-\text{close}(t-n)}{\text{close}(t-n))
- **Moving Average Convergence Divergence (MACD)**: Consists of the MACD line, signal line, and the histogram. The MACD
  line is calculated with a slow period of 12 and a fast period of 26.

![MACD](https://latex.codecogs.com/svg.image?\text{MACD}(t)=\text{EMA}_{\text{fast}}(t)-\text{EMA}_{\text{slow}}(t))


-**Momentum (MOM)**:With a time period of 4. It measures the rate of rise or fall in prices.
  ![MOM](https://latex.codecogs.com/svg.image?{MOM}(t,n)=\text{close}(t)-\text{close}(t-n))

- **Relative Strength Index (RSI)**:With a time period of 10. It measures the magnitude of recent price changes to
  evaluate overbought or oversold conditions in the price of a stock or other asset.


![RSI](https://latex.codecogs.com/svg.image?{RSI}(t,n)=100-\frac{100}{1+\frac{\text{avgGain}(t,n)}{\text{avgLoss}(t,n)}})


- **Bollinger Bands (BB)**: These include the upper band, middle band, and the lower band. They are calculated with a
  time period of 20. It is used to determine the volatility of the price.


  ![BB](https://latex.codecogs.com/svg.image?\text{upperBand}(t)=\text{SMA}(t,20)+2\times\text{rollingStd}(t,20))
  ![BB](https://latex.codecogs.com/svg.image?\text{middleBand}(t)=\text{SMA}(t,20))
  ![BB](https://latex.codecogs.com/svg.image?\text{lowerBand}(t)=\text{SMA}(t,20)-2\times\text{rollingStd}(t,20))


- **Commodity Channel Index (CCI)**: With a time period of 20. It measures the difference between the current price and
  the average price over a given time period.


  ![CCI](https://latex.codecogs.com/svg.image?\text{CCI}(t,n)=\frac{\text{MTP}(t)-\text{SMA&space;of&space;MTP}(t,n)}{0.015\times\text{mean&space;deviation&space;of&space;MTP&space;from&space;its&space;SMA&space;over}n\text{periods}})


- **Stochastic Oscillator**: The 'slowk' line is used. It is calculated with a time period of 14.

  ![Stochastic Oscillator](https://latex.codecogs.com/svg.image?\text{slowk}(t)=\frac{\text{close}(t)-\text{low}(t,n)}{\text{high}(t,n)-\text{low}(t,n)})

- **Average True Range (ATR)**: With a time period of 14. It measures the volatility of a stock or other "`security`".

  ![ATR](https://latex.codecogs.com/svg.image?\text{ATR}(t,n)=\frac{\text{ATR}(t-1,n)\times(n-1)&plus;\text{TR}(t)}{n})

- **On-Balance Volume (OBV)**: With a time period of 10. It measures the positive and negative flow of volume in a
  "`security`" relative to its price over time.

  ![OBV](https://latex.codecogs.com/svg.image?\text{OBV}(t)=\text{OBV}(t-1)&plus;\begin{cases}\text{volume}(t)&\text{if}\text{close}(t)>\text{close}(t-1)\\-\text{volume}(t)&\text{if}\text{close}(t)<\text{close}(t-1)\\0&\text{if}\text{close}(t)=\text{close}(t-1)\end{cases})


## Formula Breakdown

1. **Normalization of Indicators**: Each indicator is normalized using the Z-score formula:

![Normalized Indicator Formula](https://latex.codecogs.com/gif.latex?%5Ctext%7Bnormalized%5C_indicator%7D%20%3D%20%5Cfrac%7B%5Ctext%7Bindicator%7D%20-%20%5Ctext%7Brolling%5C_mean%28indicator%2C%20window%29%7D%7D%7B%5Ctext%7Brolling%5C_std%28indicator%2C%20window%29%7D%7D)

Where`rolling_mean` and `rolling_std` are the rolling mean and standard deviation of the indicator over a specified
window.

2. **Dynamic Weights**: The weight of momentum can increase in a strong trend. This is determined by:

![Is Strong Trend Formula](https://latex.codecogs.com/gif.latex?%5Ctext%7Bis%5C_strong%5C_trend%7D%20%3D%20%5Cleft%7C%20%5Ctext%7Bma%7D%20-%20%5Ctext%7Bclose%7D%20%5Cright%7C%20%3E%20%5Cleft%28%20%5Ctext%7Brolling%5C_mean%28trend%5C_strength%2C%20window%3D14%29%7D%20&plus;%201.5%20%5Ctimes%20%5Ctext%7Brolling%5C_std%28trend%5C_strength%2C%20window%3D14%29%7D%20%5Cright%29)

![trendStrength](https://latex.codecogs.com/svg.image?trendStrength=|ma-close|)

3. **Aggregate Score \( S \)**: It's a weighted sum of the normalized indicators:

![Aggregate Score Formula](https://latex.codecogs.com/gif.latex?S%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20w_i%20%5Ctimes%20%5Ctext%7Bnormalized%5C_indicator%7D_i)

Where `w_i` is the weight of the `i`th indicator.

4. **Market Regime Filter \( R \)**: It determines the market regime based on Bollinger Bands and a long-term moving
   average. The market can be bullish, bearish, or neutral.

5. **Volatility Adjustment \( V \)**: Adjusts the score based on market volatility. It's inversely proportional to the
   Bollinger Band width and the ATR.

6. **Final Target Score \( T \)**:
   ![Target](https://latex.codecogs.com/svg.image?&space;T=S\ast&space;R\ast&space;V&space;)

This final score **`T`** is used as the target for the AI model.

Finally, the target score is used with a threshold to determine the buy and sell signals.

## Conclusion

The LSTMRegressor model is a powerful tool that can be used to predict the outcome of a trading strategy. It can be
trained on historical data to learn the patterns and trends in the data. One of the challenges is to make sure that you're
not overfitting. There are many ways to prevent overfitting, but it's not always easy to find the right balance.
One way is use dropout layers and regularization, number of layers and neurons, and the number of epochs. Another challenge is to avoid trading on noise. This can be done by using a threshold to filter out the noise. Can be done using dissimilarity measures.
With the right
hyperparameters and the `slow` hardware that I'm using (M1 Max / RTX 3070) I was able to achieve a good results (57%
accuracy) on a small dataset of 360 days.
The model's accuracy may be improved by tuning the hyperparameters, strategy parameters \(thresholds , config params\)
using `optuna` or `keras-tuner` or other methods like grid search or random search and by using a larger dataset and
stronger hardware.
