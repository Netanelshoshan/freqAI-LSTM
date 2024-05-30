# A Trading Model Utilizing a Dynamic Weighting and Aggregate Scoring System with LSTM Networks

A [regression model](https://github.com/Netanelshoshan/freqAI-LSTM)
for  [FreqAI](https://www.freqtrade.io/en/stable/freqai/) module
from [freqtrade](https://github.com/freqtrade/freqtrade), a crypto trading platform.

⚠️ **Since problems have started to arise with the latest versions of Freqtrade (> 2024.02), I will be porting this model, and potentially other models, to PyTorch. PyTorch has better GPU support across platforms and allows for faster development since it eliminates the need to edit the core of freqAI (maybe just increasing the timeframe limit from 5 minutes to a larger one).** ⚠️


💡  **Feel free to check out the progress of the new PyTorch model in the pull request [#5](/../../pull/5) or the [pytorch](https://github.com/Netanelshoshan/freqAI-LSTM/tree/4-pytorch-migration) branch. So far It has shown >90% acc on a 1/3 of the dataset used by tensorflow. Meaning 120 days.**

**I encourage you to give it a spin, all you need to do is to copy and paste the files. Use the same config.json, add the torch config I added in my pull request and use the same example strategy. If you tackle any bugs of problems, feel free to reach out. 🙂 (Avoid using the quickstart guide below - which is for the TensorFlow model.)** 💡


## Overview

The `LSTMRegressor` is a deep learning model specifically tailored for predicting cryptocurrency prices and trends. It leverages the power of Long Short-Term Memory (LSTM) cells, a type of recurrent neural network (RNN), to understand sequential data like time series.

## Features

- **LSTM Layers**: The model utilizes multiple LSTM layers to capture sequential patterns in the data.
- **Automatic GPU Configuration**: The model can detect and configure GPU settings based on the operating system.
- **Data Preprocessing**: Robust scaling and reshaping of data for LSTM training.
- **Learning Rate Scheduler**: Adjusts the learning rate based on validation loss to optimize training.
- **TensorBoard Integration**: Allows users to monitor training progress and visualize metrics.
- **Early Stopping**: Prevents overfitting by stopping training once the validation loss stops improving.
- **Residual Connections**: Enhances the network's learning capability by connecting layers directly.

## Quick Start

1. Ensure you have tensorflow installed.
2. Make sure to set "model_save_type" in your config.json to "keras". See config-sample.json for an example.
3. In freqtrade/freqai/data_drawer.py , freqtrade/freqai/freqai_interface.py, and freqtrade/configuration/config_validation.py you should add the following code:
```python
# freqtrade/freqai/data_drawer.py
# save_data()
...
elif self.model_type == 'keras':
    model.save(save_path / f"{dk.model_filename}_model.keras")

# load_data()
...
elif self.model_type == 'keras':
    from tensorflow.keras.models import load_model
    model = load_model(dk.data_path / f"{dk.model_filename}_model.keras")

# freqtrade/freqai/freqai_interface.py : model_exists() add the following:
elif self.dd.model_type == "keras":
    file_type = ".keras"

#freqtrade/configuration/config_validation.py
...
def _validate_freqai_include_timeframes()
  ...
  if freqai_enabled:
        main_tf = conf.get('timeframe', '5m') -> change to '1h' or the min timeframe of your choosing
```
4. Download json data
```
freqtrade download-data --exchange binance --days 450 -t 4h -p ETH/USDT:USDT ETC/USDT:USDT SOL/USDT:USDT OCEAN/USDT:USDT  BTC/USDT:USDT  --data-format-ohlcv json --erase --trading-mode futures

freqtrade download-data --exchange binance --days 450 -t 1h -p ETH/USDT:USDT ETC/USDT:USDT SOL/USDT:USDT OCEAN/USDT:USDT  BTC/USDT:USDT  --data-format-ohlcv json --erase --trading-mode futures
```
5. Run it. 
```shell
freqtrade backtesting  -c config-example.json --breakdown day week month --timerange 20240301-20240401```
```
## Configuration

You can customize various parameters of the model, including:

- Number of LSTM layers (`num_lstm_layers`)
- Number of epochs for training (`epochs`)
- Batch size (`batch_size`)
- Learning rate (`learning_rate`)
- Dropout rate (`dropout_rate`)
- Timesteps (`timesteps`)

These can be adjusted through the `config.json` file.

```json
{
  ...
  "freqai": {
    "keras": true,
    "model_save_type": "keras",
    "conv_width": 24, //timesteps
    ...
    "model_training_parameters": {
      "num_lstm_layers": 3,
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.001,
      "dropout_rate": 0.3
    }
  }
}
```


# Strategy Overview

The strategy is built upon various technical indicators to determine the most optimal trading decisions. The core of the
strategy is a scoring
system, which combines these indicators to produce a single score, denoted by **`T`**, for each time point in the data.

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

```python
  # Step 0: Calculate new indicators
  df['ma'] = ta.SMA(df, timeperiod=10)
  df['roc'] = ta.ROC(df, timeperiod=2)
  df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'], slowperiod=12,
                                                        fastperiod=26)
  df['momentum'] = ta.MOM(df, timeperiod=4)
  df['rsi'] = ta.RSI(df, timeperiod=10)
  bollinger = ta.BBANDS(df, timeperiod=20)
  df['bb_upperband'] = bollinger['upperband']
  df['bb_middleband'] = bollinger['middleband']
  df['bb_lowerband'] = bollinger['lowerband']
  df['cci'] = ta.CCI(df, timeperiod=20)
  df['stoch'] = ta.STOCH(df)['slowk']
  df['atr'] = ta.ATR(df, timeperiod=14)
  df['obv'] = ta.OBV(df)
  
  # Step 1: Normalize Indicators
  df['normalized_stoch'] = (df['stoch'] - df['stoch'].rolling(window=14).mean()) / df[
      'stoch'].rolling(window=14).std()
  df['normalized_atr'] = (df['atr'] - df['atr'].rolling(window=14).mean()) / df[
      'atr'].rolling(window=14).std()
  df['normalized_obv'] = (df['obv'] - df['obv'].rolling(window=14).mean()) / df[
      'obv'].rolling(window=14).std()
  df['normalized_ma'] = (df['close'] - df['close'].rolling(window=10).mean()) / df[
      'close'].rolling(window=10).std()
  df['normalized_macd'] = (df['macd'] - df['macd'].rolling(window=26).mean()) / df[
      'macd'].rolling(window=26).std()
  df['normalized_roc'] = (df['roc'] - df['roc'].rolling(window=2).mean()) / df[
      'roc'].rolling(window=2).std()
  df['normalized_momentum'] = (df['momentum'] - df['momentum'].rolling(window=4).mean()) / \
                              df['momentum'].rolling(window=4).std()
  df['normalized_rsi'] = (df['rsi'] - df['rsi'].rolling(window=10).mean()) / df[
      'rsi'].rolling(window=10).std()
  df['normalized_bb_width'] = (df['bb_upperband'] - df['bb_lowerband']).rolling(
      window=20).mean() / (df['bb_upperband'] - df['bb_lowerband']).rolling(window=20).std()
  df['normalized_cci'] = (df['cci'] - df['cci'].rolling(window=20).mean()) / df[
      'cci'].rolling(window=20).std()
  
  # Step 1.5: Calculate momentum weight
  # Dynamic Weights (The following is an example. Increase the weight of momentum in a strong trend)
  trend_strength = abs(df['ma'] - df['close'])
  strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(
      window=14).std()
  is_strong_trend = trend_strength > strong_trend_threshold
  df['w_momentum'] = np.where(is_strong_trend, self.rsi_w.value * 1.5, self.rsi_w.value)
  
  # Step 2: Calculate Target Score S
  # Each weight should be between 0 and 1. The sum of all weights should be 1. 
  # The higher the weight, the more important the indicator.
  w = [self.ma_w.value, self.macd_w.value, self.roc_w.value, self.rsi_w.value, self.bb_w.value, self.cci_w.value,
       self.obv_w.value, self.atr_w.value, self.stoch_w.value]
  df['S'] = w[0] * df['normalized_ma'] + w[1] * df['normalized_macd'] + w[2] * df[
      'normalized_roc'] + w[3] * df['normalized_rsi'] + w[4] * \
            df['normalized_bb_width'] + w[5] * df['normalized_cci'] + df['w_momentum'] * df['normalized_momentum'] + self.stoch_w.value * df[
      'normalized_stoch'] + self.atr_w.value * df['normalized_atr'] + self.obv_w.value * df[
                 'normalized_obv']
  
  # Step 3: Calculate Market Regime Filter R
  df['R'] = 0
  df.loc[(df['close'] > df['bb_middleband']) & (
          df['close'] > df['bb_upperband']), 'R'] = 1
  df.loc[(df['close'] < df['bb_middleband']) & (
          df['close'] < df['bb_lowerband']), 'R'] = -1
  
  # Step 3.5: Additional Market Regime Filter based on long-term MA
  df['ma_100'] = ta.SMA(df, timeperiod=100)
  df['R2'] = np.where(df['close'] > df['ma_100'], 1, -1)
  
  # Step 4: Calculate Volatility Adjustment V
  bb_width = (df['bb_upperband'] - df['bb_lowerband']) / df['bb_middleband']
  df['V'] = 1 / bb_width  # assuming V is inversely proportional to BB width
  
  # New Volatility Adjustment using ATR
  df['V2'] = 1 / df['atr']
  
  # Step 5: Calculate the target score T
  df['T'] = df['S'] * df['R'] * df['V'] * df['R2'] * df['V2']
```

## Technical Indicators

The strategy employs the following technical indicators with their respective time periods:

- **Simple Moving Average (SMA)**: with a time period of 10. It is the unweighted mean of the previous `n` data points.
  ![SMA](https://latex.codecogs.com/svg.image?{SMA}(t,n)=\frac{1}{n}\sum_{i=0}^{n-1}\text{close}(t-i))

- **Rate of Change (ROC)**: With a time period of 2. It measures the percentage change in price between the current  price and the price `n` periods ago.
  ![ROC](https://latex.codecogs.com/svg.image?{ROC}(t,n)=\frac{\text{close}(t)-\text{close}(t-n)}{\text{close}(t-n))

- **Moving Average Convergence Divergence (MACD)**: Consists of the MACD line, signal line, and the histogram. The MACD  line is calculated with a slow period of 12 and a fast period of 26.
  ![MACD](https://latex.codecogs.com/svg.image?\text{MACD}(t)=\text{EMA}_{\text{fast}}(t)-\text{EMA}_{\text{slow}}(t))

- **Momentum (MOM)**:With a time period of 4. It measures the rate of rise or fall in prices.
  ![MOM](https://latex.codecogs.com/svg.image?{MOM}(t,n)=\text{close}(t)-\text{close}(t-n))


- **Relative Strength Index (RSI)**:With a time period of 10. It measures the magnitude of recent price changes to  evaluate overbought or oversold conditions in the price of a stock or other asset.
  ![RSI](https://latex.codecogs.com/svg.image?{RSI}(t,n)=100-\frac{100}{1+\frac{\text{avgGain}(t,n)}{\text{avgLoss}(t,n)}})

- **Bollinger Bands (BB)**: These include the upper band, middle band, and the lower band. They are calculated with a time period of 20. It is used to determine the volatility of the price.

  ![BB](https://latex.codecogs.com/svg.image?\text{upperBand}(t)=\text{SMA}(t,20)+2\times\text{rollingStd}(t,20))

  ![BB](https://latex.codecogs.com/svg.image?\text{middleBand}(t)=\text{SMA}(t,20))

  ![BB](https://latex.codecogs.com/svg.image?\text{lowerBand}(t)=\text{SMA}(t,20)-2\times\text{rollingStd}(t,20))


- **Commodity Channel Index (CCI)**: With a time period of 20. It measures the difference between the current price and
  the average price over a given time period.
  ![CCI](https://latex.codecogs.com/svg.image?\text{CCI}(t,n)=\frac{\text{MTP}(t)-\text{SMA&space;of&space;MTP}(t,n)}{0.015\times\text{mean&space;deviation&space;of&space;MTP&space;from&space;its&space;SMA&space;over}n\text{periods}})

- **Stochastic Oscillator**: The 'slowk' line is used. It is calculated with a time period of 14.
  ![Stochastic Oscillator](https://latex.codecogs.com/svg.image?\text{slowk}(t)=\frac{\text{close}(t)-\text{low}(t,n)}{\text{high}(t,n)-\text{low}(t,n)})

- **Average True Range (ATR)**: With a time period of 14. It measures the volatility of a stock or other "`security`.

  ![ATR](https://latex.codecogs.com/svg.image?\text{ATR}(t,n)=\frac{\text{ATR}(t-1,n)\times(n-1)&plus;\text{TR}(t)}{n})

- **On-Balance Volume (OBV)**: With a time period of 10. It measures the positive and negative flow of volume in a "`security`" relative to its price over time.

  ![OBV](https://latex.codecogs.com/svg.image?\text{OBV}(t)=\text{OBV}(t-1)&plus;\text{volume}(t)\times\text{sign}(\text{close}(t)-\text{close}(t-1)))


## Challenges and Future Improvements

One of the challenges in using the LSTMRegressor model is to ensure that the model is not overfitting. Overfitting can be mitigated by using techniques such as dropout layers, regularization, adjusting the number of layers and neurons, and tuning the number of epochs.

Another challenge is to avoid trading on noise. This can be addressed by using a threshold to filter out the noise or by employing dissimilarity measures.

With the right hyperparameters and the slow hardware (M1 Max / RTX 3070), the model achieved an accuracy of 70.4% on a small dataset of 360 days. The model's performance can be further improved by:

- Tuning the hyperparameters and strategy parameters (thresholds, config params) using techniques like `optuna`, `keras-tuner`, grid search, or random search.
- Using a larger dataset for training.
- Utilizing stronger hardware resources.

## Contributing

Contributions to the project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/netanelshoshan/freqAI-LSTM).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The LSTMRegressor model is built using the [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) libraries.
- The strategy is implemented using the [Freqtrade](https://www.freqtrade.io/) framework.
