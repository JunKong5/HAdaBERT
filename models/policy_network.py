import torch
import math
# import torch.nn as nn
from torch import nn
# import torch.nn.init as init
from torch.autograd import Variable
from transformers.modeling_bert import (
    BertEncoder,
    BertModel,
    BertPreTrainedModel,
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertConfig
)
BertLayerNorm = torch.nn.LayerNorm






class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model = 768, dropout = 0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.embedding_dim = 768
        self.hidden_size = 768
        self.num_labels = 12*2
        self.embeddings = BertEmbeddings(config)

        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.hidden_size,
                          bidirectional=False,
                          batch_first=True)

        self.position = PositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def init_embedding_layer(self, pre_trained_embedding_layer):
        self.embeddings.load_state_dict(pre_trained_embedding_layer.state_dict())


    def forward(self, x):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)
        embeds = self.embeddings(x)
        pembedding = self.position(embeds)

        gru_outs, _ = self.rnn(pembedding)
        embed = self.transformer_encoder(gru_outs)  # 768

        max_embedding, _ = torch.max(embed, dim=1)
        logits = self.classifier(max_embedding)
        return logits


