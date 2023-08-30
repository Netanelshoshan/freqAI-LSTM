import numpy as np
import tensorflow as tf


class TimeseriesGenerator:
    """
      A class for generating batches of temporal data for training, validation, and testing.

      Attributes:
          input_width (int): Number of time steps in each input sample.
          label_width (int): Number of time steps in each output label.
          shift (int): Number of time steps to shift the labels in the time dimension.
          batch_size (int): Number of samples per batch.
          shuffle (bool): Whether to shuffle the data before batching.
          train_data (tf.data.Dataset): Training dataset.
          val_data (tf.data.Dataset): Validation dataset.
          test_data (tf.data.Dataset): Test dataset.
      """

    def __init__(
            self,
            input_width,
            label_width,
            shift,
            train_df=None,
            val_df=None,
            test_df=None,
            train_labels=None,
            val_labels=None,
            test_labels=None,
            batch_size=32,
            shuffle=False,
    ):

        """
                Initializes the TimeseriesGenerator object with training, validation, and test data.

                Args:
                    input_width (int): Number of time steps in each input sample.
                    label_width (int): Number of time steps in each output label.
                    shift (int): Number of time steps to shift the labels in the time dimension.
                    train_df (pd.DataFrame, optional): Training data. Defaults to None.
                    val_df (pd.DataFrame, optional): Validation data. Defaults to None.
                    test_df (pd.DataFrame, optional): Test data. Defaults to None.
                    train_labels (pd.DataFrame, optional): Training labels. Defaults to None.
                    val_labels (pd.DataFrame, optional): Validation labels. Defaults to None.
                    test_labels (pd.DataFrame, optional): Test labels. Defaults to None.
                    batch_size (int, optional): Number of samples per batch. Defaults to 32.
                    shuffle (bool, optional): Whether to shuffle the data before batching. Defaults to False.
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_data = self._create_dataset(train_df, train_labels)
        self.val_data = self._create_dataset(val_df, val_labels)
        self.test_data = self._create_dataset(test_df, test_labels)

    def _create_dataset(self, data, labels):
        """
        Creates a tf.data.Dataset object from the given data and labels.

        Args:
            data (pd.DataFrame): Input data.
            labels (pd.DataFrame): Output labels.

        Returns:
            tf.data.Dataset: Dataset object containing the data and labels.
        """
        if data is None:
            return None

        data = np.array(data, dtype=np.float32)
        targets = None

        if labels is not None:
            if len(labels) != len(data):
                raise ValueError("Data and labels must have the same length")
            targets = np.array(labels, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=self.input_width + self.shift,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        if self.label_width:
            ds = ds.map(lambda x, y: (x[:, :self.input_width], y[:, -self.label_width:]))

        return ds

    @property
    def train(self):
        """
        Returns the training dataset.

        Returns:
            tf.data.Dataset: Training dataset.
        """
        return self.train_data

    @property
    def val(self):
        """
        Returns the validation dataset.

        Returns:
            tf.data.Dataset: Validation dataset.
        """
        return self.val_data

    @property
    def test(self):
        """
        Returns the test dataset.

        Returns:
            tf.data.Dataset: Test dataset.
        """
        return self.test_data

    @property
    def inference(self):
        """
        Returns the test dataset for inference.

        Returns:
            tf.data.Dataset: Test dataset for inference.
        """
        return self.test_data.map(lambda x: x)
