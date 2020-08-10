import torch
import torch.nn as nn
import torch.nn.functional as F
from model.top_down_attention import GatedMLP
class Classifier(nn.Module):
    def __init__(self, in_dimensions, out_dimensions):
        super(Classifier, self).__init__()
        self.gated_MLP = GatedMLP(in_dimensions, 2*in_dimensions, bias=True)
        self.linear = nn.Linear(2*in_dimensions, out_dimensions, bias=True)

    def forward(self, input):
        x = self.gated_MLP(input)
        return torch.sigmoid(self.linear(x))