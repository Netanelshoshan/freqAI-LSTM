## Quick Start - TensorFlow

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