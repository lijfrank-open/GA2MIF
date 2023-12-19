import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MLP(torch.nn.Module):
    def __init__(self, hidesize, list_mlp, dropout, activation=''):
        super(MLP, self).__init__()
        self.list_mlp = list_mlp
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        mlp = []
        dro = []
        self.mlp0 = nn.Linear(hidesize,list_mlp[0])
        self.dropout0 = nn.Dropout(dropout)
        for l in range(len(self.list_mlp)-1):
            mlp.append(nn.Linear(self.list_mlp[l],list_mlp[l+1]))
            dro.append(nn.Dropout(dropout))
        self.mlps = nn.ModuleList(mlp)
        self.dropouts = nn.ModuleList(dro)

    def forward(self, emotions_f):
        x = emotions_f
        x = self.mlp0(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout0(x)
        for l in range(len(self.list_mlp)-1):
            x = self.mlps[l](x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.dropouts[l](x)
        return x