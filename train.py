import torch
import torch.nn as nn
from dataset import Dataset
from vqa_model import VQAModel
def train():
    model = VQAModel(18248, 300)
    optim = torch.optim.Adam(model.grad_params(), 0.001)
    dataset = Dataset('', 'train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    device = torch.device("cuda:0")
    model.to(device)
    for f, bb, spat, obs, a, q, q_len, qid in loader:
        f, bb, spat, obs, a, q, q_len, qid = f.to(device), bb.to(device), spat.to(device), \
                                             obs.to(device), a.to(device), q.to(device), q_len.to(device), qid.to(device)
                                             
        optim.zero_grad()
        pred=model(q,q_len, f,obs)
        loss = compute_multi_loss(pred, a)
        print(loss)
        loss_item =  round(loss.item(),3)
        loss.backward()
        nn.utils.clip_grad_norm_(model.grad_params(), 5)
        optim.step()

        
        

def compute_multi_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).to(labels.device)
    loss *= labels.size(1)
    return loss

train()
