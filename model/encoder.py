import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, emb_dim, h_dim, n_layers, dropout, max_len = 14):
        super(Encoder, self).__init__()
        self.max_len = max_len
        self.encoder = nn.LSTM(emb_dim, h_dim, n_layers, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, in_lengths):
        embeddings = embeddings.permute(1,0,2)
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths=in_lengths,enforce_sorted=False)

        self.encoder.flatten_parameters()
        packed_outputs, (hidden, cell) = self.encoder(embeddings)

        outputs, lengths = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=self.max_len)
        outputs = outputs.permute(1,0,2)
        return cell[-1], hidden[-1], outputs
