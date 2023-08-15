import logging
import platform
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as spy
import tensorflow as tf
from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, RepeatVector, Permute, Multiply, Add
from keras.layers import Dense, Input, LSTM, Flatten, TimeDistributed, Dropout
from pandas import DataFrame
from tensorflow_addons.optimizers import AdamW

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

logger = logging.getLogger(__name__)


class LSTMRegressor(BaseRegressionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        config = self.freqai_info.get("model_training_parameters", {})

        # Determine current operating system
        system = platform.system()

        # Set GPU configuration based on OS
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if system == "Windows":
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=7900)]
                )
            elif system == "Linux":
                # Adjust memory limit as needed for Linux systems
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
                )
            elif system == "Darwin":
                logger.info(
                    "MacOS is detected. There's a big impact on performance running on gpu,"
                    " consider uninstalling tensorflow-metal")
        else:
            logger.info("No GPU found. The model will run on CPU.")

        self.num_lstm_layers = config.get('num_lstm_layers', 3)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 64)
        self.learning_rate: float = config.get("learning_rate", 0.0001)
        self.dropout_rate = config.get("dropout_rate", 0.3)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:

        n_features = data_dictionary["train_features"].shape[1]
        n_output = data_dictionary["train_labels"].shape[1]

        # Reshape input to be 3D [samples, timestamps, features]
        train_X = data_dictionary["train_features"].values.reshape(
            data_dictionary["train_features"].shape[0], 1, n_features)
        train_y = data_dictionary["train_labels"].values

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            test_X = data_dictionary["test_features"].values.reshape(
                data_dictionary["test_features"].shape[0], 1, n_features)
            test_y = data_dictionary["test_labels"].values
        else:
            test_X, test_y = None, None

        # Designing the model
        d_model = n_features
        num_lstm_layers = self.num_lstm_layers
        epochs = self.epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        dropout_rate = self.dropout_rate
        input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))
        x = input_layer

        # Add more LSTM layers with attention
        for _ in range(num_lstm_layers):
            lstm_output = LSTM(d_model, return_sequences=True, dropout=dropout_rate)(x)

            # Add attention mechanism to LSTM hidden states
            attention = Dense(1, activation='tanh')(lstm_output)
            attention = Flatten()(attention)
            attention = Activation('softmax')(attention)
            attention = RepeatVector(d_model)(attention)
            attention = Permute([2, 1])(attention)
            attention_output = Multiply()([lstm_output, attention])
            x = Add()([lstm_output, attention_output])

        x = TimeDistributed(Dense(d_model, activation='tanh'))(x)
        x = Dropout(dropout_rate)(x)

        x = Flatten()(x)
        x = Dense(d_model, activation='relu')(x)

        # Add residual connection
        x = Add()([x, input_layer])
        output_layer = Dense(n_output)(x)
        model = Model(inputs=input_layer, outputs=output_layer)

        # Optimizer
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.001)
        model.compile(loss='mae', optimizer=optimizer)

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001)
        tensorboard_callback = TensorBoard(log_dir=dk.data_path, histogram_freq=1)

        # Fit network
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                  validation_data=(test_X, test_y), verbose=2, shuffle=False,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                             lr_scheduler, tensorboard_callback])

        return model

    def predict(
            self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Generate predictions using the fitted TransformerRegressor model.

        :param unfiltered_df: Full dataframe for the current backtest period.
        :param dk: FreqaiDataKitchen object for the current coin/model.
        :return: Tuple containing a DataFrame of predictions and a numpy array of indicators (1s and 0s) marking
                 places where data was removed (NaNs) or where model was uncertain (PCA and DI index).
        """

        dk.find_features(unfiltered_df)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"], outliers, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True)

        input_data = np.expand_dims(dk.data_dictionary["prediction_features"], axis=1)
        predictions = self.model.predict(input_data)
        if self.CONV_WIDTH == 1:
            predictions = np.reshape(predictions, (-1, len(dk.label_list)))

        pred_df = DataFrame(predictions, columns=dk.label_list)

        pred_df, _, _ = dk.label_pipeline.inverse_transform(pred_df)
        if dk.feature_pipeline["di"]:
            dk.DI_values = dk.feature_pipeline["di"].di_values
        else:
            dk.DI_values = np.zeros(outliers.shape[0])
        dk.do_predict = outliers

        return pred_df, dk.do_predict
