import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable




class AttnGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttnGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

        init.xavier_normal_(self.Wr.state_dict()['weight'])
        init.xavier_normal_(self.Ur.state_dict()['weight'])
        init.xavier_normal_(self.W.state_dict()['weight'])
        init.xavier_normal_(self.U.state_dict()['weight'])

    def forward(self, c, hi_1, g):


        r_i = torch.sigmoid(self.Wr(c) + self.Ur(hi_1))

        h_tilda = torch.tanh(self.W(c) + r_i * self.U(hi_1))
        hi = g * h_tilda + (1 - g) * hi_1


        return hi



class AttGRU(nn.Module):
    def __init__(self, bidirectional=True):
        super(AttGRU, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = AttnGRUCell(768, 768)
        if self.bidirectional:
            self.rnn_bwd = AttnGRUCell(768, 768)

    def forward(self, context, init_hidden, att_score, attn_mask=None):


        hidden = init_hidden
        if self.bidirectional:
            hidden, hidden_bwd = init_hidden.unsqueeze(1).transpose(0, 1).contiguous(), init_hidden.unsqueeze(1).transpose(0, 1).contiguous()
        inp = context.transpose(0, 1).contiguous()
        gates = att_score.unsqueeze(1)
        gates = gates.transpose(1, 2).transpose(0, 1).contiguous()



        seq_len = context.size()[1]

        for i in range(seq_len):
            hidden = self.rnn(inp[i:i + 1], hidden, gates[i:i + 1])

            if self.bidirectional:
                hidden_bwd = self.rnn_bwd(inp[seq_len - i - 1:seq_len - i], hidden_bwd,
                                          gates[seq_len - i - 1:seq_len - i])

        output = hidden.transpose(0, 1).contiguous()

        if self.bidirectional:
            output = torch.cat([hidden, hidden_bwd], dim=-1).transpose(0, 1).contiguous()  # batch x 1 x d_h*2
        return output





class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.pooling_linear = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        weights = self.pooling_linear(x).squeeze(dim=2)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        att_score = nn.Softmax(dim=-1)(weights)

        return att_score


class AGMencoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.num_classes = config.num_labels
        self.hops = 1
        self.hidden_size = 768
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2, batch_first=True, bidirectional=True)
        self.att = SelfAttention(input_dim=768)
        self.AttGRUs = AttGRU()
        self.dropout_mid = nn.Dropout(0.3)
        self.liner = nn.Linear(768*2,200)

        self.classifier = nn.Linear(200, self.num_classes)

    def init_hidden(self,  batch_size, d_model):
        return Variable(torch.zeros(batch_size, d_model)).to("cuda:3")

    def forward(self, embeddings, mask):
        if not hasattr(self, '_flattened'):
            self.lstm.flatten_parameters()
            setattr(self, '_flattened', True)
        hidden_state = self.dropout_mid(self.lstm(embeddings)[0])

        hidden_states = hidden_state+embeddings


        att_score = self.att(hidden_states,mask)

        s_out = []
        for hop in range(self.hops):
            attn_hid = self.init_hidden(embeddings.size(0),embeddings.size(-1))

            out_put = self.AttGRUs(hidden_states,attn_hid,att_score,mask)

            s_out.append(out_put)

        s_cont = torch.cat(s_out, dim=-1).squeeze(1)

        s_cont=self.dropout_mid(nn.ReLU()(self.liner(s_cont)))
        logits = self.classifier(s_cont)
        return logits





