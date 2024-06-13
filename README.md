[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# A Trading Model Utilizing a Dynamic Weighting and Aggregate Scoring System with LSTM Networks

A regression model and trading strategy for  [FreqAI](https://www.freqtrade.io/en/stable/freqai/) module
from [freqtrade](https://github.com/freqtrade/freqtrade), a crypto trading bot.


⚠️ **Since problems have started to arise with the latest versions of Freqtrade (> 2024.02), I will be porting this
model, and potentially other models, to PyTorch. PyTorch has better GPU support across platforms and allows for faster
development since it eliminates the need to edit the core of freqAI (maybe just increasing the timeframe limit from 5
minutes to a bigger one).** ⚠️

## Overview

This project aims to develop a trading model that utilizes a dynamic weighting and aggregate scoring system to make more informed trading decisions. The model was initially built using TensorFlow and the Keras API, but has been ported to PyTorch to take advantage of its better GPU support across platforms and faster development process.

## Quick Start

1. Clone the repository

```shell
git clone https://github.com/Netanelshoshan/freqAI-LSTM.git
```
2. Copy the files to the freqtrade directory

```shell 
cp torch/BasePyTorchModel.py <freqtrade dir>/freqtrade/freqai/base_models/
cp torch/PyTorchLSTMModel.py <freqtrade dir >/freqtrade/freqai/torch/
cp torch/PyTorchModelTrainer.py <freqtrade dir>/freqtrade/freqai/torch/
cp torch/PyTorchLSTMRegressor.py <freqtrade dir>/user_data/freqaimodels/
cp config-example.json <freqtrade dir>/user_data/config.json
cp ExampleLSTMStrategy.py <freqtrade dir>/user_data/strategies/
```
3. Download the data
```shell
freqtrade download-data -c user_data/config-torch.json --timerange 20230101-20240529 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase
```
4. Edit "freqtrade/configuration/config_validation.py"
```python
...
def _validate_freqai_include_timeframes()
...
    if freqai_enabled:
        main_tf = conf.get('timeframe', '5m') -> change to '1h' or the **min** timeframe of your choosing
```
5. Make sure your package is edible after the the changes
```shell
pip install -e .
```

7. Run the backtest
```shell
freqtrade backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240301-20240401 
````

## Quick Start with docker

1. Clone the repository

```shell
git clone https://github.com/Netanelshoshan/freqAI-LSTM.git
```
2. Build local docker images

```shell
cd freqAI-LSTM
docker build -f torch/Dockerfile  -t freqai .
```
3. Download data and Run the backtest
```
docker run -v ./data:/freqtrade/user_data/data  -it freqai  download-data -c user_data/config-torch.json --timerange 20230101-20240529 --timeframe 15m 30m 1h 2h 4h 8h 1d --erase

docker run -v ./data:/freqtrade/user_data/data  -it freqai  backtesting -c user_data/config-torch.json --breakdown day week month --timerange 20240301-20240401 
```




## Model Architecture

The core of the model is a Long Short-Term Memory (LSTM) network, which is a type of recurrent neural network that excels at handling sequential data and capturing long-term dependencies.

The LSTM model (PyTorchLSTMModel) has the following architecture:

1. The input data is passed through a series of LSTM layers (the number of layers is configurable via the `num_lstm_layers` parameter.). Each LSTM layer is followed by a Batch Normalization layer and a Dropout layer for regularization.
2. The output from the last LSTM layer is then passed through a fully connected layer with ReLU activation.
3. An Alpha Dropout layer is applied for additional regularization.
4. Finally, the output is passed through another fully connected layer to produce the final predictions.

The model's hyperparameters, such as the number of LSTM layers, hidden dimensions, dropout rates, and others, can be easily configured through the `model_kwargs` parameter in the `model_training_parameters` section of the configuration file.

Here's an example of how the model_training_parameters can be set up:

```json
"model_training_parameters": {
    "learning_rate": 3e-3,
    "trainer_kwargs": {
    "n_steps": null,
    "batch_size": 32,
    "n_epochs": 10,
    },
    "model_kwargs": {
    "num_lstm_layers": 3,
    "hidden_dim": 128,
    "dropout_percent": 0.4,
    "window_size": 5
    }
}
```
Let's go through each of these parameters:

- `learning_rate`: This is the learning rate used by the optimizer during training. It controls the step size at which the model's weights are updated in response to the estimated error each time the model weights are updated.
- `trainer_kwargs`: These are keyword arguments passed to the `PyTorchLSTMTrainer` which is located in PyTorchModelTrainer.
    - `n_steps`: The number of training iterations. If set to null, the number of epochs (n_epochs) will be used instead.
    - `batch_size`: The number of samples per gradient update.
    -  `n_epochs`: The number of times to iterate over the dataset.
- `model_kwargs`: These are keyword arguments passed to the `PyTorchLSTMModel`.
    - `num_lstm_layers`: The number of LSTM layers in the model.
    - `hidden_dim`: The dimensionality of the output space (i.e., the number of hidden units) in each LSTM layer.
    - `dropout_percent`: The dropout rate for regularization. Dropout is a technique used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
    - `window_size`: The number of time steps (or data points in the above case) to look back when making predictions.


## The Strategy

At its core, this strategy is all about making smart trading decisions by looking at the market from different angles. It's like having a team of experts, each focusing on a specific aspect of the market, and then combining their insights to make a well-informed decision.

Here's how it works:

1. **Indicators**: The strategy calculates a bunch of technical indicators, which are like different lenses to view the market. These indicators help identify trends, momentum, volatility, and other important market characteristics.

2. **Normalization**: To make sure all the indicators are on the same page. it normalizes them by calculating the z-score. This step ensures that the indicators are comparable and can be weighted appropriately.

3. **Dynamic Weighting**: The strategy is adaptable and can adjust the importance of different indicators based on market conditions.

4. **Aggregate Score**: All the normalized indicators are combined into a single score, which represents the overall market sentiment. Just like taking a vote among the experts to reach a consensus.

5. **Market Regime Filters**: The strategy considers the current market regime, whether it's bullish, bearish, or neutral. Looking up the weather before deciding on an outfit. 🌞🌧️?

6. **Volatility Adjustments**: It takes into account the market's volatility and adjusts the target score accordingly. We want to be cautious when the market is choppy and more aggressive when it's calm.

7. **Final Target Score**: All these factors are combined into a final target score, which is like a concise and informative signal for the LSTM model to learn from. It's like giving the model a clear and focused task to work on.

8. **Entry and Exit Signals**: we use the predicted target score and set thresholds to determine when to enter or exit a trade.

## Why It Works

Using a multi-factor target score allows the strategy to consider multiple aspects of the market simultaneously, leading to more robust and informed decision-making.

By reducing noise and focusing on the most relevant information, the target score helps the LSTM model learn from a cleaner and more meaningful signal, filtering out the distractions and focusing on what really matters.

The dynamic weighting and market regime filters make the strategy adaptable to changing market conditions. We want the strategy to `"think"` and adjust to new situations.


```python
# Step 0: Calculate new indicators
dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                            fastperiod=26)
dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
bollinger = ta.BBANDS(dataframe, timeperiod=20)
dataframe['bb_upperband'] = bollinger['upperband']
dataframe['bb_middleband'] = bollinger['middleband']
dataframe['bb_lowerband'] = bollinger['lowerband']
dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
dataframe['obv'] = ta.OBV(dataframe)

# Step 1: Normalize Indicators:
# Why? Normalizing the indicators will make them comparable and allow us to assign weights to them.
# How? We will calculate the z-score of each indicator by subtracting the rolling mean and dividing by the
# rolling standard deviation. This will give us a normalized value that is centered around 0 with a standard
# deviation of 1.
dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe[
    'stoch'].rolling(window=14).std()
dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe[
    'atr'].rolling(window=14).std()
dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe[
    'obv'].rolling(window=14).std()
dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe[
    'close'].rolling(window=10).std()
dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe[
    'macd'].rolling(window=26).std()
dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe[
    'roc'].rolling(window=2).std()
dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                   dataframe['momentum'].rolling(window=4).std()
dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe[
    'rsi'].rolling(window=10).std()
dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
    window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe[
    'cci'].rolling(window=20).std()

# Dynamic Weights (Example: Increase the weight of momentum in a strong trend)
trend_strength = abs(dataframe['ma'] - dataframe['close'])

# Calculate the rolling mean and standard deviation of the trend strength to determine a strong trend
# The threshold is set to 1.5 times the standard deviation above the mean, but can be adjusted as needed
strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(
    window=14).std()

# Assign a higher weight to momentum if the trend is strong
is_strong_trend = trend_strength > strong_trend_threshold

# Assign the dynamic weights to the dataframe
dataframe['w_momentum'] = np.where(is_strong_trend, self.w3.value * 1.5, self.w3.value)

# Step 2: Calculate aggregate score S
w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value,
     self.w6.value, self.w7.value, self.w8.value]

dataframe['S'] = w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + w[2] * dataframe[
    'normalized_roc'] + w[3] * dataframe['normalized_rsi'] + w[4] * \
                 dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + dataframe[
                     'w_momentum'] * dataframe['normalized_momentum'] + self.w8.value * dataframe[
                     'normalized_stoch'] + self.w7.value * dataframe['normalized_atr'] + self.w6.value * \
                 dataframe['normalized_obv']

# Step 3: Market Regime Filter R
# EXPLANATION: If the price is above the upper Bollinger Band, assign a value
# of 1 to R. If the price is below the lower Bollinger Band, assign a value of -1 to R. Otherwise,
# the value R stays 0.
# What's basically happening here is that we are assigning a value of 1 to R when
# the price is in the upper band, -1 when the price is in the lower band, and 0 when the price is in the
# middle band. This is a simple way to determine the market regime based on Bollinger Bands. What is market
# regime? Market regime is the state of the market. It can be trending, ranging, or reversing. So we are
# using Bollinger Bands to determine the market regime. You can use other indicators to determine the market
# regime as well. For example, you can use moving averages, RSI, MACD, etc.
dataframe['R'] = 0
dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (
        dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (
        dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

# Additional Market Regime Filter based on long-term MA
dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)

# Step 4: Volatility Adjustment V
# EXPLANATION: Calculate the Bollinger Band width and assign it to V. The Bollinger Band width is the
# difference between the upper and lower Bollinger Bands divided by the middle Bollinger Band. The idea is
# that when the Bollinger Bands are wide, the market is volatile, and when the Bollinger Bands are narrow,
# the market is less volatile. So we are using the Bollinger Band width as a measure of volatility. You can
# use other indicators to measure volatility as well. For example, you can use the ATR (Average True Range)
bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
dataframe['V'] = 1 / bb_width  # example, assuming V is inversely proportional to BB width

# Another Volatility Adjustment using ATR
dataframe['V2'] = 1 / dataframe['atr']

# Get Final Target Score to incorporate new calculations
dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['V'] * dataframe['R2'] * dataframe['V2']

# Assign the target score T to the AI target column
dataframe['&-target'] = dataframe['T']
```

## Putting It All Together

In a nutshell, by calculating and normalizing indicators, applying dynamic weighting, considering market regimes, adjusting for volatility, and using a multi-factor target score, the strategy provides a comprehensive and efficient signal for the LSTM model to learn from.

It's a powerful combination of technical analysis, adaptability, and deep learning that aims to navigate the market effectively and make profitable trading decisions.


## Challenges and Future Improvements

One of the challenges are ensuring model is not overfitting. We mitigated that using dropout layers, regularization, adjusting the number of layers and neurons, and tuning the number of epochs.

Another challenge is to avoid trading on noise. This can be addressed by using a threshold and weights, to filter out the noise or by employing dissimilarity measures.

With the right hyperparameters and the hardware like M1 Max / RTX3070 , the model achieved an accuracy of >90.0% on a small dataset of 120 days on backtesting using minimal config and trying hard to avoid overfitting.

![](https://private-user-images.githubusercontent.com/45298885/335341423-b54c065b-0429-485b-9c70-2184f12692cd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTczMzExNjYsIm5iZiI6MTcxNzMzMDg2NiwicGF0aCI6Ii80NTI5ODg4NS8zMzUzNDE0MjMtYjU0YzA2NWItMDQyOS00ODViLTljNzAtMjE4NGYxMjY5MmNkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjAyVDEyMjEwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdjMzg0OTNjZGRjOTk0N2M4MzE5OTgyZjhkMTM2ZDViYWVjZTY5MjRjZWNmZjY5ODI0YTZmMjQxM2FhN2Q5MTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.NFmDQ1xFgWZvejd9lVsyI_afSoYx3dOY9rjUd0ASC2g)

Backtest result on two pairs, with the new and improved PyTorch model.


![](https://private-user-images.githubusercontent.com/45298885/335342184-3b27d994-3bc3-4ea1-ba68-e252a0a03aa2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTczMzExNjYsIm5iZiI6MTcxNzMzMDg2NiwicGF0aCI6Ii80NTI5ODg4NS8zMzUzNDIxODQtM2IyN2Q5OTQtM2JjMy00ZWExLWJhNjgtZTI1MmEwYTAzYWEyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA2MDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNjAyVDEyMjEwNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc0NGQyZDFlNTIyZjkzMThhM2M4MjczY2ExOTRkOTg3NGMxMWM5YTJlODFhMjU3MWI3ZjVlMWRlODZlMTk4M2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.tx5qNJawxJYRYNrttHh4Vf_5vK_qle019s7TOBMblOI)

Daily returns on two pairs over march 2024. The model is fairly strict and doesn't generate a lot of signals. 

## Contributing

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an
issue or submit a pull request on the [GitHub repository](https://github.com/netanelshoshan/freqAI-LSTM).


