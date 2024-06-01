import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class PyTorchLSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model implemented using PyTorch.

    This class serves as a complex example for the integration of PyTorch models.
    It is designed to handle sequential data and capture long-term dependencies.

    :param input_dim: The number of input features. This parameter specifies the number
        of features in the input data that the LSTM will use to make predictions.
    :param output_dim: The number of output classes. This parameter specifies the number
        of classes that the LSTM will predict.
    :param hidden_dim: The number of hidden units in each LSTM layer. This parameter controls
        the complexity of the LSTM and determines how many nonlinear relationships the LSTM
        can represent. Increasing the number of hidden units can increase the capacity of
        the LSTM to model complex patterns, but it also increases the risk of overfitting
        the training data. Default: 100
    :param dropout_percent: The dropout rate for regularization. This parameter specifies
        the probability of dropping out a neuron during training to prevent overfitting.
        The dropout rate should be tuned carefully to balance between underfitting and
        overfitting. Default: 0.3
    :param n_layer: The number of LSTM layers. This parameter specifies the number
        of LSTM layers in the model. Adding more layers can increase its capacity to
        model complex patterns, but it also increases the risk of overfitting
        the training data. Default: 1

    :returns: The output of the LSTM, with shape (batch_size, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        self.num_lstm_layers: int = kwargs.get("num_lstm_layers", 1)
        self.hidden_dim: int = kwargs.get("hidden_dim", 100)
        self.dropout_percent: float = kwargs.get("dropout_percent", 0.3)

        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.lstm_layers.append(nn.LSTM(input_dim, self.hidden_dim, batch_first=True))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        self.dropouts.append(nn.Dropout(p=self.dropout_percent))

        if self.num_lstm_layers > 1:
            for _ in range(self.num_lstm_layers - 1):
                self.lstm_layers.append(nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True))
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
                self.dropouts.append(nn.Dropout(p=self.dropout_percent))

        self.fc1 = nn.Linear(self.hidden_dim, 36)
        self.alpha_dropout = nn.AlphaDropout(p=0.5)
        self.fc2 = nn.Linear(36, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor x is of shape (batch_size, sequence_length, input_dim)
        if x.dim() == 2:
            # If the input is (batch_size, input_dim), add a dummy sequence dimension
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        for i in range(self.num_lstm_layers):
            x, _ = self.lstm_layers[i](x)
            if x.dim() == 3:
                x = self.batch_norms[i](x[:, -1, :])  # Apply batch norm on the last output from the LSTM
            else:
                x = self.batch_norms[i](x)  # Apply batch norm directly if x has 2 dimensions
            x = self.dropouts[i](x)
            if i > 0:
                x = x + x_res
            x_res = x

        x = self.relu(self.fc1(x))
        x = self.alpha_dropout(x)
        x = self.fc2(x)
        return x
