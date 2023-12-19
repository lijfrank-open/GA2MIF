import torch
from model_gcn import GATv2Conv_Layer
import torch.nn as nn
from torch.nn import functional as F

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

class MDGAT(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers, hidesize):
        super(MDGAT, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
    def forward(self, features, edge_index):
        output = features
        for mod in self.layers:
            output = mod(output, edge_index)
        return output

class MDGATLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, num_heads=5, agg_type = '', use_residual=True, activation='', norm_first=False, no_cuda=False):
        super(MDGATLayer, self).__init__()
        self.no_cuda = no_cuda
        if agg_type == '' or 'None':
            self.agg_type = None
        else:
            self.agg_type = agg_type
        self.hidesize = hidesize
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(hidesize)

        if self.agg_type is None:
            self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=True, concat=False)
        else:
            self.convs = GATv2Conv_Layer(hidesize, hidesize, heads=num_heads, add_self_loops=False, concat=False)

        if self.agg_type == 'sum-product1':
            self.sumpro2 = nn.Linear(2*hidesize, hidesize)
        elif self.agg_type == 'sum-product2':
            self.sumpro = nn.Linear(hidesize, hidesize)
            self.sumpro1 = nn.Linear(hidesize, hidesize)
            self.sumpro2 = nn.Linear(2*hidesize, hidesize)
        elif self.agg_type == 'sum':
            self.sumpro = nn.Linear(hidesize, hidesize)
            self.sumpro1 = nn.Linear(hidesize, hidesize)
        elif self.agg_type == 'concat':
            self.sumpro2 = nn.Linear(2*hidesize, hidesize)
        elif self.agg_type is None:
            pass
        else:
            raise RuntimeError("not {}".format(agg_type))
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        self.norm_first = norm_first

    def forward(self, features, edge_index):
        x = features
        if self.norm_first:
            if self.use_residual:
                x = x + self.graph_conv(self.norm(x), edge_index)
            else:
                x = self.graph_conv(self.norm(x), edge_index)
        else:
            if self.use_residual:
                x = self.norm(x + self.graph_conv(x, edge_index))
            else:
                x = self.norm(self.graph_conv(x, edge_index))
        return x

    def graph_conv(self, x, edge_index):
        if self.agg_type is None:
            x = self.convs(x, edge_index)
        else:
            con_f = self.convs(x, edge_index)
        if self.agg_type == 'sum-product1':
            O1 = x + con_f
            O2 = x * con_f
            x = self.sumpro2(torch.cat((O1, O2), dim=-1))
        elif self.agg_type == 'sum-product2':
            O1 = self.sumpro(x + con_f)
            O2 = self.sumpro1(x * con_f)
            x = self.sumpro2(torch.cat((O1, O2), dim=-1))
        elif self.agg_type == 'sum':
            x = self.sumpro(x) + self.sumpro1(con_f)
        elif self.agg_type == 'concat':
            x = self.sumpro2(torch.cat((x, con_f), dim=-1))
        elif self.agg_type is None:
            pass
        return x