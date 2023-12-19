import torch
import copy
import torch.nn as nn
from model_utils import pad
from torch.nn import functional as F

class MPCAT(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(MPCAT, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    def forward(self, features_a, features_v, features_l, key_padding_mask): 
        for mod in self.layers:
            features_a, features_v, features_l = mod(features_a, features_v, features_l, key_padding_mask)
        return features_a, features_v, features_l

class MPCATLayer(torch.nn.Module):
    def __init__(self, feature_size, nheads=4, dropout=0.3, use_residual=True, modals='', no_cuda=False):
        super(MPCATLayer, self).__init__()  
        self.no_cuda = no_cuda
        self.use_residual = use_residual
        self.modals = modals
        if 'a' in self.modals:
            self.multihead_attn_a = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.multihead_attn_a1 = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.dropout_a = nn.Dropout(dropout)
            self.dropout_a1 = nn.Dropout(dropout)
            self.dropout_a2 = nn.Dropout(dropout)
            self.norm_a = nn.LayerNorm(feature_size)
            self.norm_a1 = nn.LayerNorm(feature_size)
            self.fc_a = nn.Linear(feature_size, 2*feature_size)
            self.fc_a1 = nn.Linear(2*feature_size, feature_size)
        if 'v' in self.modals:
            self.multihead_attn_v = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.multihead_attn_v1 = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.dropout_v = nn.Dropout(dropout)
            self.dropout_v1 = nn.Dropout(dropout)
            self.dropout_v2 = nn.Dropout(dropout)
            self.norm_v = nn.LayerNorm(feature_size)
            self.norm_v1 = nn.LayerNorm(feature_size)
            self.fc_v = nn.Linear(feature_size, 2*feature_size)
            self.fc_v1= nn.Linear(2*feature_size, feature_size)
        if 'l' in self.modals:
            self.multihead_attn_l = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.multihead_attn_l1 = nn.MultiheadAttention(feature_size, nheads, batch_first=True)
            self.dropout_l = nn.Dropout(dropout)
            self.dropout_l1 = nn.Dropout(dropout)
            self.dropout_l2 = nn.Dropout(dropout) 
            self.norm_l = nn.LayerNorm(feature_size)
            self.norm_l1 = nn.LayerNorm(feature_size)
            self.fc_l = nn.Linear(feature_size, 2*feature_size)
            self.fc_l1 = nn.Linear(2*feature_size, feature_size)

    def forward(self, features_a, features_v, features_l, key_padding_mask):
        att_a, att_v, att_l = self.cross_modal_att(features_a, features_v, features_l, key_padding_mask)
        if self.use_residual:
            if 'a' in self.modals:
                features_a = self.norm_a(features_a + att_a)
            if 'v' in self.modals:
                features_v = self.norm_v(features_v + att_v)
            if 'l' in self.modals:
                features_l = self.norm_l(features_l + att_l)
        else:
            if 'a' in self.modals:
                features_a = self.norm_a(att_a)
            if 'v' in self.modals:
                features_v = self.norm_v(att_v)
            if 'l' in self.modals:
                features_l = self.norm_l(att_l)
        
        full_a, full_v, full_l = self.fullcon(features_a, features_v, features_l)
        if self.use_residual:
            if 'a' in self.modals:
                features_a = self.norm_a1(features_a + full_a)
            if 'v' in self.modals:
                features_v = self.norm_v1(features_v + full_v)
            if 'l' in self.modals:
                features_l = self.norm_l1(features_l + full_l) 
        else:
            features_a = self.norm_a1(full_a)
            features_v = self.norm_v1(full_v)
            features_l = self.norm_l1(full_l)
        return features_a, features_v, features_l
    
    def cross_modal_att(self, features_a, features_v, features_l, key_padding_mask):
        if 'a' in self.modals and 'v' in self.modals:
            att_v, attn_output_weights_v = self.multihead_attn_v(features_a, features_v, features_v, key_padding_mask=key_padding_mask)
        if 'a' in self.modals and 'l' in self.modals:
            att_l, attn_output_weights_l = self.multihead_attn_l(features_a, features_l, features_l, key_padding_mask=key_padding_mask)
        if 'v' in self.modals and 'a' in self.modals:
            att_a, attn_output_weights_a = self.multihead_attn_a(features_v, features_a, features_a, key_padding_mask=key_padding_mask)
        if 'v' in self.modals and 'l' in self.modals:
            att_l1, attn_output_weights_l1 = self.multihead_attn_l1(features_v, features_l, features_l, key_padding_mask=key_padding_mask)
        if 'l' in self.modals and 'a' in self.modals:
            att_a1, attn_output_weights_a1 = self.multihead_attn_a1(features_l, features_a, features_a, key_padding_mask=key_padding_mask)
        if 'l' in self.modals and 'v' in self.modals:
            att_v1, attn_output_weights_v1 = self.multihead_attn_v1(features_l, features_v, features_v, key_padding_mask=key_padding_mask)
        if len(self.modals)==3:
            att_a = att_a + att_a1
            att_v = att_v + att_v1
            att_l = att_l + att_l1
        if 'a' in self.modals and 'v' in self.modals:
            att_a = att_a
            att_v = att_v
        if 'a' in self.modals and 'l' in self.modals:
            att_a = att_a1
            att_l = att_l
        if 'l' in self.modals and 'v' in self.modals:
            att_v = att_v1
            att_l = att_l1    

        if 'a' in self.modals:
            att_a = self.dropout_a(att_a)
        else:
            att_a = []
        if 'v' in self.modals:
            att_v = self.dropout_v(att_v)
        else:
            att_v = []
        if 'l' in self.modals:
            att_l = self.dropout_l(att_l)
        else:
            att_l = []
        return att_a, att_v, att_l

    def fullcon(self, features_a, features_v, features_l):
        if 'a' in self.modals:
            features_a = self.dropout_a2(self.fc_a1(self.dropout_a1(F.relu(self.fc_a(features_a)))))
        if 'v' in self.modals:
            features_v = self.dropout_v2(self.fc_v1(self.dropout_v1(F.relu(self.fc_v(features_v)))))
        if 'l' in self.modals:
            features_l = self.dropout_l2(self.fc_l1(self.dropout_l1(F.relu(self.fc_l(features_l)))))
        return features_a, features_v, features_l

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])