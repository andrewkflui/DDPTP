import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

#codes are heavily modified from https://github.com/xuehaouwa/poppl
#### Attention definition ####
class Attention(nn.Module):
    """
    use batch_first = False mode
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        h = hidden[-1].repeat(max_len, 1, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_energies = self.score(h, encoder_outputs)  
        return F.softmax(attn_energies, dim=1)  

    def score(self, hidden, encoder_outputs):

        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1) 
        energy = torch.bmm(v, energy)

        return energy.squeeze(1)


class Attention_general(nn.Module):
    """
    use batch_first = False mode
    """

    def __init__(self, hidden_dim):
        super(Attention_general, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, hidden_dim)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_energies = self.score(hidden, encoder_outputs)  
        return F.softmax(attn_energies, dim=1)  

    def score(self, hidden, encoder_outputs):

        energy = F.tanh(self.attn(encoder_outputs))
        energy = energy.permute(1, 2, 0) 
        hidden = hidden.permute(1, 0, 2)
        energy = torch.bmm(hidden, energy)  

        return energy.squeeze(1) 

#### Classification submodel definition #### 
    
class Classnet(nn.Module):
    def __init__(self,input_dim,LSTM_hidden_dim,LSTM_num_layers = 1,dropout_rate = 0.5 ,CNN_kerenl_size=3,CNN_out_channel=64,pooling_size =2 ,RC_class_size=6):
        super().__init__()
        self.llstm = nn.LSTM(input_dim,LSTM_hidden_dim,LSTM_num_layers,batch_first = True)
        self.rlstm = nn.LSTM(input_dim,LSTM_hidden_dim,LSTM_num_layers,batch_first = True)
        self.lstm = nn.LSTM(64,LSTM_hidden_dim,LSTM_num_layers,batch_first =True)
        self.dropout = nn.Dropout()
        self.conv1d0 = nn.Conv1d(LSTM_hidden_dim,CNN_out_channel,CNN_kerenl_size,stride=1)
        self.conv1d1 = nn.Conv1d(64,CNN_out_channel,CNN_kerenl_size,stride=1)
        self.pooling1d0 = nn.MaxPool1d(pooling_size)
        self.bnorm0 = nn.BatchNorm1d(LSTM_hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(64)
        self.bnorm2 = nn.BatchNorm1d(128)
        self.bnorm3 = nn.BatchNorm1d(RC_class_size)
        self.leftdense = nn.Linear(input_dim,input_dim)
        self.rightdense = nn.Linear(input_dim,input_dim)
        self.outputdense = nn.Linear(128,RC_class_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        rx = torch.flip(x,[1])
        lx = self.leftdense(x)
        lx = self.activation(lx)
        
        rx = self.rightdense(rx)
        rx = self.activation(rx)
        lx,_ = self.llstm(lx)
        rx,_ = self.rlstm(rx)
        
        x = torch.stack([lx,rx])
        x = torch.sum(x, (0))
        x = x.permute((0,2,1))
        x = self.dropout(x)
        x = self.conv1d0(x)
        x = self.activation(x)
        x = self.pooling1d0(x)
        x = self.bnorm1(x)
        x = self.conv1d1(x)
        x = self.activation(x)
        x = self.pooling1d0(x)  
        x = self.dropout(x)
        x = x.permute((0,2,1))
        
        x,_ = self.lstm(x)
        x = x.permute((0,2,1))
        x = self.bnorm2(x)
        x = x.permute((0,2,1))
        x = x[:,-1]
        x = self.outputdense(x)
        x = self.bnorm3(x)
        x = self.softmax(x)
        return x    
    
#### Recurrent submodel definitions ####
class Baseline_Plain_GRU_model(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len):
        super(Baseline_Plain_GRU_model, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(2, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.gru_loc_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.out2loc = nn.Linear(hidden_dim, 2)

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        loc_encoder_outputs = []

        emb_loc = self.loc_embedding(obs)
        emb_loc = self.tanh(emb_loc)
        
        self.gru_loc.flatten_parameters()
        
        enc_out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)        
        out_loc = self.out2loc(enc_out_loc)

        
        out_loc = out_loc[:,-2:-1,:]
        # decoding 
        for _ in range(self.pred_len):
            out_loc = self.loc_embedding(out_loc)
            out_loc = self.tanh(out_loc)
            out_loc,hidden_loc = self.gru_loc(out_loc, hidden_loc)
            out_loc = self.out2loc(out_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))
        out = torch.stack(predicted, dim=1)
        
        return out

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.cuda()
    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
        
class Baseline_Enc_Dec_model(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len,input_dim=2):
        super(Baseline_Enc_Dec_model, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(input_dim, embedding_dim)
        self.loc_embedding_dec = nn.Linear(2, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.gru_loc_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.gru_loc_dec = nn.GRU(embedding_dim , hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.out2loc = nn.Linear(hidden_dim, 2)

    def predict_one_step(self, last_loc, enc_out_loc, hidden_loc):
        loc_embedded = self.loc_embedding_dec(last_loc)
        loc_embedded = self.tanh(loc_embedded)
        
        dec_out_loc, hidden_loc = self.gru_loc_dec(loc_embedded, hidden_loc)
        dec_out_loc, hidden_loc = self.gru_loc_dec_2(dec_out_loc, hidden_loc)    
        
        self.gru_loc_dec.flatten_parameters()
        self.gru_loc_dec_2.flatten_parameters()
        
        out_loc = self.out2loc(dec_out_loc)
        return hidden_loc, out_loc

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        loc_encoder_outputs = []

        emb_loc = self.loc_embedding(obs)
        emb_loc = self.tanh(emb_loc)
        
        self.gru_loc.flatten_parameters()
        self.gru_loc_2.flatten_parameters()
        
        enc_out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)
        enc_out_loc, hidden_loc = self.gru_loc_2(enc_out_loc, hidden_loc)

        
        loc_encoder_outputs.append(enc_out_loc)
        out_loc = self.out2loc(enc_out_loc)
        loc_encoder_outputs = torch.cat(loc_encoder_outputs, dim=1)
        
        out_loc = out_loc[:,-2:-1,:]
        # decoding 
        for _ in range(self.pred_len):
            hidden_loc, out_loc = self.predict_one_step(out_loc, loc_encoder_outputs, hidden_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))
        out = torch.stack(predicted, dim=1)
        
        return out

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.cuda()
    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
        

class GRU_submodel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len, general=False,v2=False,enc_input_dim=2,dec_input_dim=2):
        super(GRU_submodel, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(enc_input_dim, embedding_dim)
        self.loc_embedding_dec = nn.Linear(dec_input_dim, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.gru_loc_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.gru_loc_dec = nn.GRU(embedding_dim , hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout,bidirectional =False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        if general:
            self.temporal_attention_loc = Attention_general(hidden_dim=hidden_dim)
        else:
            self.temporal_attention_loc = Attention(hidden_dim=hidden_dim)
        self.out2loc = nn.Linear(hidden_dim, 2)
        self.v2 = v2

    def predict_one_step(self, last_loc, enc_out_loc, hidden_loc):
        loc_embedded = self.loc_embedding_dec(last_loc)
        loc_embedded = self.tanh(loc_embedded)

        dec_out_loc, hidden_loc = self.gru_loc_dec(loc_embedded, hidden_loc)

        dec_out_loc, hidden_loc = self.gru_loc_dec_2(dec_out_loc, hidden_loc)
        
        self.gru_loc_dec.flatten_parameters()
        out_loc = self.out2loc(dec_out_loc)

        return hidden_loc, out_loc

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        loc_encoder_outputs = []

        emb_loc = self.loc_embedding(obs)
        emb_loc = self.tanh(emb_loc)
        
        self.gru_loc.flatten_parameters()
        self.gru_loc_2.flatten_parameters()
        if self.v2:
            enc_out_loc, _ = self.gru_loc(emb_loc, hidden_loc)
            enc_out_loc, _ = self.gru_loc_2(enc_out_loc, hidden_loc)
        else:
            enc_out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)
            enc_out_loc, hidden_loc = self.gru_loc_2(enc_out_loc, hidden_loc)
        loc_encoder_outputs.append(enc_out_loc)
        out_loc = self.out2loc(enc_out_loc)
        loc_encoder_outputs = torch.cat(loc_encoder_outputs, dim=1)
        
        out_loc = out_loc[:,-2:-1,:]
        # decoding 
        for _ in range(self.pred_len):
            hidden_loc, out_loc = self.predict_one_step(out_loc, loc_encoder_outputs, hidden_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))
        out = torch.stack(predicted, dim=1)
        
        return out

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.cuda()
    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
class GRU_Attention_submodel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len, general=False,enc_input_dim=2,dec_input_dim=2):
        super(GRU_Attention_submodel, self).__init__()
        self.enc_input_dim = enc_input_dim
        self.dec_input_dim = dec_input_dim
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(enc_input_dim, embedding_dim)
        self.loc_embedding_dec = nn.Linear(dec_input_dim, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        if general:
            self.temporal_attention_loc = Attention_general(hidden_dim=hidden_dim)
        else:
            self.temporal_attention_loc = Attention(hidden_dim=hidden_dim)
        self.out2loc = nn.Linear(hidden_dim, 2)

    def predict_one_step(self, last_loc, enc_out_loc, hidden_loc):
        if self.enc_input_dim == self.dec_input_dim:
            loc_embedded = self.loc_embedding(last_loc)
        else:
            loc_embedded = self.loc_embedding_dec(last_loc)
        loc_embedded = self.tanh(loc_embedded)
        
        temporal_weight_loc = self.temporal_attention_loc(hidden_loc, enc_out_loc)
        context_loc = temporal_weight_loc.unsqueeze(1).bmm(enc_out_loc)
        emb_con_loc = torch.cat((loc_embedded, context_loc), dim=2)

        dec_out_loc, hidden_loc = self.gru_loc_dec(emb_con_loc, hidden_loc)
        dec_out_loc, hidden_loc = self.gru_loc_dec_2(dec_out_loc, hidden_loc)
        
        self.gru_loc_dec.flatten_parameters()
        self.gru_loc_dec_2.flatten_parameters()
        
        out_loc = self.out2loc(dec_out_loc)

        return hidden_loc, out_loc

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        obs_loc = torch.index_select(obs, dim=2, index=self.generate_index([0, 1], use_gpu=True))
        loc_encoder_outputs = []

        emb_loc = self.loc_embedding(obs)
        emb_loc = self.tanh(emb_loc)
        
        self.gru_loc.flatten_parameters()
        
        enc_out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)

        
        loc_encoder_outputs.append(enc_out_loc)
        out_loc = self.out2loc(enc_out_loc)
        loc_encoder_outputs = torch.cat(loc_encoder_outputs, dim=1)
        out_loc = out_loc[:,-2:-1,:]
        # decoding 
        for _ in range(self.pred_len):
            hidden_loc, out_loc = self.predict_one_step(out_loc, loc_encoder_outputs, hidden_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))
        out = torch.stack(predicted, dim=1)

        return out

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.cuda()
    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
        
class BiGRU_submodel(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, pred_len, general=False,enc_input_dim=2,dec_input_dim=2):
        super(BiGRU_model, self).__init__()
        self.enc_input_dim = enc_input_dim
        self.dec_input_dim = dec_input_dim
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.loc_embedding = nn.Linear(enc_input_dim, embedding_dim)
        self.loc_embedding_dec = nn.Linear(dec_input_dim, embedding_dim)
        self.gru_loc_left = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_right = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.gru_loc_dec_2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        if general:
            self.temporal_attention_loc = Attention_general(hidden_dim=hidden_dim)
        else:
            self.temporal_attention_loc = Attention(hidden_dim=hidden_dim)
        self.out2loc = nn.Linear(hidden_dim, 2)

    def predict_one_step(self, last_loc, enc_out_loc, hidden_loc):
        if self.dec_input_dim == self.enc_input_dim:
            loc_embedded = self.loc_embedding(last_loc)
        else:
            loc_embedded = self.loc_embedding_dec(last_loc)
        loc_embedded = self.tanh(loc_embedded)

        dec_out_loc, hidden_loc = self.gru_loc_dec(loc_embedded, hidden_loc)
        dec_out_loc, hidden_loc = self.gru_loc_dec_2(dec_out_loc, hidden_loc)
        
        self.gru_loc_dec.flatten_parameters()
        self.gru_loc_dec_2.flatten_parameters()
        
        out_loc = self.out2loc(dec_out_loc)

        return hidden_loc, out_loc

    def forward(self, obs):
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        predicted = []
        # encoding
        loc_encoder_outputs = []
        emb_loc = self.loc_embedding(obs)
        emb_loc = self.tanh(emb_loc)
        emb_loc_r = torch.flip(emb_loc,[1])
        
        self.gru_loc_left.flatten_parameters()
        self.gru_loc_right.flatten_parameters()
        self.gru_loc_2.flatten_parameters()
        
        enc_out_loc_l, _ = self.gru_loc_left(emb_loc, hidden_loc)
        enc_out_loc_r, _ = self.gru_loc_right(emb_loc_r, hidden_loc)
        enc_out_loc_l = self.relu(enc_out_loc_l)
        enc_out_loc_r = self.relu(enc_out_loc_r)
        
        emb_loc = torch.stack([enc_out_loc_l,enc_out_loc_r])
        emb_loc = torch.sum(emb_loc, (0))
        
        hidden_loc = self.init_hidden(batch_size=obs.size(0))
        enc_out_loc, hidden_loc = self.gru_loc_2(emb_loc, hidden_loc)
        loc_encoder_outputs.append(enc_out_loc)

        out_loc = self.out2loc(enc_out_loc)
        loc_encoder_outputs = torch.cat(loc_encoder_outputs, dim=1)
        out_loc = out_loc[:,-2:-1,:]
        # decoding 
        for _ in range(self.pred_len):
            hidden_loc, out_loc = self.predict_one_step(out_loc, loc_encoder_outputs, hidden_loc)
            predicted.append(torch.squeeze(out_loc, dim=1))
        out = torch.stack(predicted, dim=1)
        return out

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h.cuda()
    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))
        