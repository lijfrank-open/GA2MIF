import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dens import DensNet, DensNetLayer
from parsers import args
from model_mdgat import MDGAT, MDGATLayer
from model_mpcat import MPCAT, MPCATLayer
from model_mlp import MLP
from model_utils import batch_graphify, simple_batch_graphify, batch_to_all, all_to_batch

cuda = torch.cuda.is_available() and not args.no_cuda

class GNNModel(nn.Module):

    def __init__(self, args, D_m_a, D_m_v, D_m, num_speakers, n_classes):
        
        super(GNNModel, self).__init__()

        self.base_model = args.base_model
        self.no_cuda = args.no_cuda
        self.num_speakers = num_speakers
        self.dropout = args.dropout
        self.modals = [x for x in args.modals]
        self.list_mlp = args.list_mlp
        self.multi_modal = args.multi_modal
        self.window_past = args.windowp
        self.window_future = args.windowf
        n_relations = 2 * num_speakers ** 2
        self.ratio_speaker = args.ratio_speaker
        self.ratio_modal = args.ratio_modal

        if 'a' in self.modals:
            if self.base_model[0] == 'LSTM':
                self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
                self.rnn_audio = nn.LSTM(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[0] == 'GRU':
                self.linear_audio = nn.Linear(D_m_a, args.base_size[0])
                self.rnn_audio = nn.GRU(input_size=args.base_size[0], hidden_size=args.hidesize, num_layers=args.base_nlayers[0], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_audio_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[0] == 'Transformer':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
                encoder_layer_audio = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_audio = nn.TransformerEncoder(encoder_layer_audio, num_layers=args.base_nlayers[0])
            elif self.base_model[0] == 'Dens':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
                encoder_layer_audio = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_audio = DensNet(encoder_layer_audio, num_layers=args.base_nlayers[0])
            elif self.base_model[0] == 'None':
                self.linear_audio = nn.Linear(D_m_a, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if 'v' in self.modals:
            if self.base_model[1] == 'LSTM':
                self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
                self.rnn_visual = nn.LSTM(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[1] == 'GRU':
                self.linear_visual = nn.Linear(D_m_v, args.base_size[1])
                self.rnn_visual = nn.GRU(input_size=args.base_size[1], hidden_size=args.hidesize, num_layers=args.base_nlayers[1], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_visual_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[1] == 'Transformer':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
                encoder_layer_visual = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_visual = nn.TransformerEncoder(encoder_layer_visual, num_layers=args.base_nlayers[1])
            elif self.base_model[1] == 'Dens':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
                encoder_layer_visual = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_visual = DensNet(encoder_layer_visual, num_layers=args.base_nlayers[1])
            elif self.base_model[1] == 'None':
                self.linear_visual = nn.Linear(D_m_v, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if 'l' in self.modals:
            if self.base_model[2] == 'LSTM':
                self.linear_text = nn.Linear(D_m, args.base_size[2])
                self.rnn_text = nn.LSTM(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[2] == 'GRU':
                self.linear_text = nn.Linear(D_m, args.base_size[2])
                self.rnn_text = nn.GRU(input_size=args.base_size[2], hidden_size=args.hidesize, num_layers=args.base_nlayers[2], batch_first=True, dropout=args.dropout, bidirectional=True)
                self.linear_text_ = nn.Linear(2*args.hidesize, args.hidesize)
            elif self.base_model[2] == 'Transformer':
                self.linear_text = nn.Linear(D_m, args.hidesize)
                encoder_layer_text = nn.TransformerEncoderLayer(d_model=args.hidesize, nhead=4, dropout=args.dropout, batch_first=True)
                self.transformer_encoder_text = nn.TransformerEncoder(encoder_layer_text, num_layers=args.base_nlayers[2])
            elif self.base_model[2] == 'Dens':
                self.linear_text = nn.Linear(D_m, args.hidesize)
                encoder_layer_text = DensNetLayer(hidesize=args.hidesize, dropout=self.dropout, activation='tanh', no_cuda=self.no_cuda)
                self.dens_text = DensNet(encoder_layer_text, num_layers=args.base_nlayers[2])
            elif self.base_model[2] == 'None':
                self.linear_text = nn.Linear(D_m, args.hidesize)
            else:
                print ('Base model must be one of .')
                raise NotImplementedError 

        if args.ratio_speaker > 0:
            self.speaker_embeddings = nn.Embedding(num_speakers, args.hidesize)
        self.rel_embeddings = nn.Embedding(n_relations, args.hidesize)

        if args.ratio_modal > 0:
            self.modal_embeddings = nn.Embedding(3, args.hidesize)
        
        if 'a' in self.modals:
            mdgatlayer_a = MDGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.list_nheads[0], agg_type = args.agg_type[0], use_residual=args.list_residual[0], activation='relu', norm_first=False, no_cuda=args.no_cuda)
            self.mdgat_a = MDGAT(mdgatlayer_a, num_layers=args.unimodal_nlayers[0], hidesize=args.hidesize)

        if 'v' in self.modals:
            mdgatlayer_v = MDGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.list_nheads[0], agg_type = args.agg_type[1], use_residual=args.list_residual[0], activation='relu', norm_first=False, no_cuda=args.no_cuda)
            self.mdgat_v = MDGAT(mdgatlayer_v, num_layers=args.unimodal_nlayers[1], hidesize=args.hidesize)

        if 'l' in self.modals:
            mdgatlayer_l = MDGATLayer(args.hidesize, dropout=args.dropout, num_heads=args.list_nheads[0], agg_type = args.agg_type[2], use_residual=args.list_residual[0], activation='relu', norm_first=False, no_cuda=args.no_cuda)
            self.mdgat_l = MDGAT(mdgatlayer_l, num_layers=args.unimodal_nlayers[2], hidesize=args.hidesize)
            
        if args.list_mlp != []:
            self.mlp_a = MLP(args.hidesize, args.list_mlp, args.dropout)
            self.mlp_v = MLP(args.hidesize, args.list_mlp, args.dropout)
            self.mlp_l = MLP(args.hidesize, args.list_mlp, args.dropout)
            self.smax_fc_a = nn.Linear(self.list_mlp[-1], n_classes)
            self.smax_fc_v = nn.Linear(self.list_mlp[-1], n_classes)
            self.smax_fc_l = nn.Linear(self.list_mlp[-1], n_classes)
        else:
            if 'a' in self.modals:
                self.smax_fc_a = nn.Linear(args.hidesize, n_classes)
            if 'v' in self.modals:
                self.smax_fc_v = nn.Linear(args.hidesize, n_classes)
            if 'l' in self.modals:
                self.smax_fc_l = nn.Linear(args.hidesize, n_classes)

        att_crossmodallayer = MPCATLayer(feature_size=args.hidesize, nheads=args.list_nheads[1], dropout=args.dropout, use_residual=args.list_residual[1], modals=args.modals, no_cuda=args.no_cuda)
        self.att_crossmodal = MPCAT(att_crossmodallayer, num_layers=args.crossmodal_nlayers)

        edge_type_mapping = {} 
        for j in range(num_speakers):
            for k in range(num_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

        self.fccccc = nn.Linear(args.hidesize*len(self.modals), args.hidesize)

        if self.list_mlp != []:  
            self.mlp = MLP(args.hidesize, args.list_mlp, args.dropout)
            self.smax_fc = nn.Linear(self.list_mlp[-1], n_classes)
        else:
            self.smax_fc = nn.Linear(args.hidesize, n_classes)

    def forward(self, U, qmask, umask, seq_lengths, max_seq_length, U_a=None, U_v=None):
        if 'a' in self.modals:
            if self.base_model[0] == 'LSTM':
                U_a = self.linear_audio(U_a)
                U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False) 
                self.rnn_audio.flatten_parameters()
                emotions_a, hidden_a = self.rnn_audio(U_a)
                emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True) 
                emotions_a = self.linear_audio_(emotions_a)
            elif self.base_model[0] == 'GRU':
                U_a = self.linear_audio(U_a)
                U_a = nn.utils.rnn.pack_padded_sequence(U_a, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_audio.flatten_parameters()
                emotions_a, hidden_a = self.rnn_audio(U_a)
                emotions_a, _ = nn.utils.rnn.pad_packed_sequence(emotions_a, batch_first=True) 
                emotions_a = self.linear_audio_(emotions_a)
            elif self.base_model[0] == 'Transformer':
                U_a = self.linear_audio(U_a)
                emotions_a = self.transformer_encoder_audio(U_a, src_key_padding_mask=umask)
            elif self.base_model[0] == 'Dens':
                U_a = self.linear_audio(U_a)
                emotions_a = self.dens_audio(U_a)
            elif self.base_model[0] == 'None':
                emotions_a = torch.tanh(self.linear_audio(U_a))
        
        if 'v' in self.modals:
            if self.base_model[1] == 'LSTM':
                U_v = self.linear_visual(U_v)
                U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_visual.flatten_parameters()
                emotions_v, hidden_v = self.rnn_visual(U_v)
                emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
                emotions_v = self.linear_visual_(emotions_v)
            elif self.base_model[1] == 'GRU':
                U_v = self.linear_visual(U_v)
                U_v = nn.utils.rnn.pack_padded_sequence(U_v, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_visual.flatten_parameters()
                emotions_v, hidden_v = self.rnn_visual(U_v)
                emotions_v, _ = nn.utils.rnn.pad_packed_sequence(emotions_v, batch_first=True)
                emotions_v = self.linear_visual_(emotions_v)
            elif self.base_model[1] == 'Transformer':
                U_v = self.linear_visual(U_v)
                emotions_v = self.transformer_encoder_visual(U_v, src_key_padding_mask=umask)
            elif self.base_model[1] == 'Dens':
                U_v = self.linear_visual(U_v)
                emotions_v = self.dens_visual(U_v)
            elif self.base_model[1] == 'None':
                emotions_v = torch.tanh(self.linear_visual(U_v))
        
        if 'l' in self.modals:
            if self.base_model[2] == 'LSTM':
                U = self.linear_text(U)
                U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_text.flatten_parameters()
                emotions_l, hidden_l = self.rnn_text(U)
                emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True)
                emotions_l = self.linear_text_(emotions_l)
            elif self.base_model[2] == 'GRU':
                U = self.linear_text(U)
                U = nn.utils.rnn.pack_padded_sequence(U, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                self.rnn_text.flatten_parameters()
                emotions_l, hidden_l = self.rnn_text(U)
                emotions_l, _ = nn.utils.rnn.pad_packed_sequence(emotions_l, batch_first=True)
                emotions_l = self.linear_text_(emotions_l)
            elif self.base_model[2] == 'Transformer':
                U = self.linear_text(U)
                emotions_l = self.transformer_encoder_text(U, src_key_padding_mask=umask)
            elif self.base_model[2] == 'Dens':
                U = self.linear_text(U)
                emotions_l = self.dens_text(U)
            elif self.base_model[2] == 'None':
                emotions_l = torch.tanh(self.linear_text(U))
    
        if 'a' in self.modals:
            features_a, edge_index, edge_type, edge_index_lengths = batch_graphify(emotions_a, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
        else:
            features_a = []
        if 'v' in self.modals:
            features_v, edge_index, edge_type, edge_index_lengths = batch_graphify(emotions_v, qmask, seq_lengths, self.window_past, self.window_future, self.edge_type_mapping, self.no_cuda)
        else:
            features_v = []
        if 'l' in self.modals:
            features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
        else:
            features_l = []

        if self.ratio_speaker > 0:
            qmask_ = torch.cat([qmask[i,:x,:] for i,x in enumerate(seq_lengths)],dim=0)
            spk_idx = torch.argmax(qmask_, dim=-1).cuda() if not self.no_cuda else torch.argmax(qmask_, dim=-1)
            spk_emb_vector = self.speaker_embeddings(spk_idx)
            if 'a' in self.modals:
                features_a = features_a + self.ratio_speaker*spk_emb_vector
            if 'v' in self.modals:
                features_v = features_v + self.ratio_speaker*spk_emb_vector
            if 'l' in self.modals:
                features_l = features_l + self.ratio_speaker*spk_emb_vector

        if self.ratio_modal > 0:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)
            if 'a' in self.modals:
                features_a = features_a + self.ratio_modal*emb_vector[0].reshape(1, -1).expand(features_a.shape[0], features_a.shape[1])
            if 'v' in self.modals:
                features_v = features_v + self.ratio_modal*emb_vector[1].reshape(1, -1).expand(features_v.shape[0], features_v.shape[1])
            if 'l' in self.modals:
                features_l = features_l + self.ratio_modal*emb_vector[2].reshape(1, -1).expand(features_l.shape[0], features_l.shape[1])

        if 'a' in self.modals:
            emotions_a = self.mdgat_a(features_a, edge_index)
        else:
            emotions_a = []

        if 'v' in self.modals:
            emotions_v = self.mdgat_v(features_v, edge_index)
        else:
            emotions_v = []
            
        if 'l' in self.modals:
            emotions_l = self.mdgat_l(features_l, edge_index)
        else:
            emotions_l = []

        if self.list_mlp != []:    
            emotions_a = self.mlp_a(emotions_a)
            emotions_v = self.mlp_v(emotions_v)
            emotions_l = self.mlp_v(emotions_l)
        if 'a' in self.modals:
            log_prob_a = self.smax_fc_a(emotions_a)
        else:
            log_prob_a = 0
        if 'v' in self.modals:
            log_prob_v = self.smax_fc_v(emotions_v)
        else:
            log_prob_v = 0
        if 'l' in self.modals:
            log_prob_l = self.smax_fc_l(emotions_l)
        else:
            log_prob_l = 0

        if 'a' in self.modals:
            emotions_a = all_to_batch(emotions_a, seq_lengths, max_seq_length, self.no_cuda)
        else:
            emotions_a = []
        if 'v' in self.modals:
            emotions_v = all_to_batch(emotions_v, seq_lengths, max_seq_length, self.no_cuda)
        else:
            emotions_v = []
        if 'l' in self.modals:
            emotions_l = all_to_batch(emotions_l, seq_lengths, max_seq_length, self.no_cuda)
        else:
            emotions_l = []
        
        emotions_a, emotions_v, emotions_l = self.att_crossmodal(emotions_a, emotions_v, emotions_l, umask)

        if 'a' in self.modals:
            emotions_a = batch_to_all(emotions_a, seq_lengths, self.no_cuda)
        else:
            emotions_a = []
        if 'v' in self.modals:
            emotions_v = batch_to_all(emotions_v, seq_lengths, self.no_cuda)
        else:
            emotions_v = []
        if 'l' in self.modals:
            emotions_l = batch_to_all(emotions_l, seq_lengths, self.no_cuda)
        else:
            emotions_l = []

        emotions = []
        if len(emotions_a) != 0:
            emotions.append(emotions_a)
        if len(emotions_v) != 0:
            emotions.append(emotions_v)
        if len(emotions_l) != 0:
            emotions.append(emotions_l)
        emotions_feat = torch.cat(emotions, dim=-1)
        emotions_feat = self.fccccc(emotions_feat)
        if self.list_mlp != []:
            emotions_feat = self.mlp(emotions_feat)

        log_prob = self.smax_fc(emotions_feat)

        if len(self.modals)==3: 
            return log_prob_a, log_prob_v, log_prob_l, log_prob
        if len(self.modals)==2:
            if 'a' in self.modals and 'v' in self.modals:
                return log_prob_a, log_prob_v, log_prob
            if 'a' in self.modals and 'l' in self.modals:
                return log_prob_a, log_prob_l, log_prob
            if 'v' in self.modals and 'l' in self.modals:
                return log_prob_v, log_prob_l, log_prob
