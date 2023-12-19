import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def _get_activation_fn(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    elif activation == "" or "None":
        return None
    raise RuntimeError("activation should be relu/gelu/tanh/sigmoid, not {}".format(activation))

class DensNet(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(DensNet, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
    def forward(self, features):
        output = features
        for mod in self.layers:
            output = mod(output)
        return output

class DensNetLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, activation='', no_cuda=False):
        super(DensNetLayer, self).__init__()
        self.norm = nn.LayerNorm(hidesize)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidesize, 2*hidesize)
        self.fc1 = nn.Linear(2*hidesize, hidesize)
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
    def forward(self, features):
        x = features
        if self.activation is None:
            x = self.norm(self.dropout1(self.fc1(self.dropout(self.fc(x)))))
        else:
            x = self.norm(self.dropout1(self.fc1(self.dropout(self.activation(self.fc(x))))))
        return x