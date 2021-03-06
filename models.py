import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchlayers as tl

class SimpleGRU(nn.Module):
    def __init__(self,
                 source_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional=False):
        """ Args:
                source_size (int): The expected number of features in the input.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
                bidirectional (boolean): whether to use bidirectional model.
        """
        super(SimpleGRU, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(source_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.fc1 = nn.Linear(num_directions*hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input, hidden=None):
        """ Args:
                input (pred_len, batch, source_size): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.

            Returns:
                output (pred_len, batch, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        # Feed source sequences into GRU:
        output, _ = self.gru(input, hidden)
        output = self.fc1(output[:, -1, :]).relu()
        output = self.fc2(output).sigmoid().unsqueeze(2)
        return output

class Encoder(nn.Module):
    def __init__(self,
                 source_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional=False):
        """ Args:
                source_size (int): The expected number of features in the input.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
                bidirectional (boolean): whether to use bidirectional model.
        """
        super(Encoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(source_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers else 0,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.compress = nn.Linear(num_layers*num_directions, num_layers)

    def forward(self, input, hidden=None):
        """ Args:
                input (pred_len, batch, source_size): Input sequence.
                hidden (num_layers*num_directions, batch, hidden_size): Initial states.

            Returns:
                output (pred_len, batch, num_directions*hidden_size): Outputs at every step.
                hidden (num_layers, batch, hidden_size): Final state.
        """
        # Feed source sequences into GRU:
        output, hidden = self.gru(input, hidden)
        # Compress bidirection to one direction for decoder:
        hidden = hidden.permute(1, 2, 0)
        hidden = self.compress(hidden)
        hidden = hidden.permute(2, 0, 1)
        return output, hidden.contiguous()

class Decoder(nn.Module):
    def __init__(self,
                 target_size,
                 hidden_size,
                 num_layers,
                 dropout):
        """ Args:
                target_size (int): The expected number of sequence features.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
        """
        super(Decoder, self).__init__()
        self.target_size = target_size
        self.gru = nn.GRU(target_size,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers else 0,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, target_size)

    def forward(self, hidden, pred_len, target=None, teacher_forcing=False):
        """ Args:
                hidden (num_layers, batch, hidden_size): States of the GRU.
                target (pred_len, batch, target_size): Target sequence. If None,
                    the output sequence is generated by feeding the output
                    of the previous timestep (teacher_forcing has to be False).
                teacher_forcing (bool): Whether to use teacher forcing or not.

            Returns:
                outputs (pred_len, batch, target_size): Tensor of log-probabilities
                    of words in the target language.
                hidden of shape (1, batch_size, hidden_size): New states of the GRU.
        """
        if target is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'

        # Determine constants:
        batch = hidden.shape[1]
        # The starting value to feed to the GRU:
        val = torch.zeros((batch, 1, self.target_size), device=hidden.device)
        if target is not None:
            target = torch.cat([val, target[:, :-1, :]], dim=1)
        # Sequence to record the predicted values:
        outputs = list()
        for i in range(pred_len):
            # Embed the value at ith time step:
            # If teacher_forcing then use the target value at current step
            # Else use the predicted value at previous step:
            val = target[:, i:i+1, :] if teacher_forcing else val
            # Feed the previous value and the hidden to the network:
            output, hidden = self.gru(val, hidden)
            # Predict new output:
            val = self.out(output.relu()).sigmoid()
            # Record the predicted value:
            outputs.append(val)
        # Concatenate predicted values:
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

class Generator(nn.Module):
    """ RNN Generator """
    def __init__(self,
                 noise_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 dropout):
        """ Args:
                noise_size (int): The expected number of features in the input.
                hidden_size (int): The number of features in the hidden state.
                output_size (int): The number of features in the output state.
                num_layers (int): Number of recurrent layers.
                dropout (float): Dropout probability.
        """
        super(Generator, self).__init__()
        self.gru = nn.GRU(noise_size+1,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, noise, condition):
        """ Args:
                noise (batch, seq_len, noise_size): Noise input sequence.
                condition (batch, seq_len, 1): Conditional input.

            Returns:
                output (batch, seq_len, output_size): Generated sequence.
        """
        batch, seq_len, _ = noise.shape
        input = torch.cat((noise, condition), dim=-1)
        output, _ = self.gru(input)
        output = self.norm(self.fc1(output)).relu()
        output = self.fc2(output).tanh()
        return output

class Discriminator(nn.Module):
    """ RNN Discriminator """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional=False):
        """ Args:
                input_size (int): The expected number of features in the input.
                hidden_size (int): The number of features in the hidden state.
                num_layers (int): Number of recurrent layers.
                dropout (float): dropout probability.
                bidirectional (boolean): whether to use bidirectional model.
        """
        super(Discriminator, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size+1,
                          hidden_size,
                          num_layers,
                          dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional,
                          batch_first=True)
        self.fc1 = nn.Linear(num_directions*hidden_size, 64)
        self.norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input, condition):
        """ Args:
                input (batch, seq_len, input_size): Input sequence.
                condition (batch, seq_len, 1): Conditional input.

            Returns:
                output (batch): Real / Fake prediction.
        """
        batch, seq_len, _ = input.shape
        input = torch.cat((input, condition), dim=-1)
        output, _ = self.gru(input)
        output = self.norm(self.fc1(output.mean(1))).relu()
        output = self.fc2(output).sigmoid().squeeze()
        return output

