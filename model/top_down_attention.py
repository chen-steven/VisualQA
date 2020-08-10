import torch
import torch.nn as nn
import torch.nn.functional as F

class TopDownAttention(nn.Module):
    def __init__(self, v_dim, q_dim, hidden_dim, dropout_p=0.2):
        super(TopDownAttention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.gated_MLP = GatedMLP(v_dim+q_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, v, q, n_objs):
        b,k,_ = v.size()
        q = q.unsqueeze(1).repeat(1,k,1)
        v_q = torch.cat((v, q),2)
        f_a = self.gated_MLP(v_q)
        att = self.attention_weights(f_a)
        return att

class GatedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GatedMLP,self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=bias)
        self.linear2 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        y = F.relu(self.linear1(input))
        g = torch.sigmoid(self.linear2(input))
        return y*g

