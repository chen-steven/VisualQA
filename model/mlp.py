import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, bn = False, ln = False, img = False,
                 classify = False,dropout = 0.2,max_objs = 100):
        super().__init__()

        self.classify = classify
        self.linear_1 = nn.Linear(in_dim, in_dim * 2)
        self.linear_2 = nn.Linear(in_dim * 2, out_dim)

        self.norm = ln or bn
        if img and bn:
           self.norm1 = nn.BatchNorm1d(max_objs)
           self.norm2 = nn.BatchNorm1d(max_objs)
        elif bn:
           self.norm1 = nn.BatchNorm1d(in_dim * 2)
           self.norm2 = nn.BatchNorm1d(out_dim)
        elif ln:
           self.norm1 = nn.LayerNorm(in_dim * 2)
           self.norm2 = nn.LayerNorm(out_dim)


        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(F.relu(self.linear_1(x))) if self.norm else self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(self.norm2(F.relu(x))) if not self.classify and self.norm else x
        #x = self.dropout(F.relu(x)) if not self.classify else x
        return x