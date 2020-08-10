import torch
import torch.nn as nn
from dataset import Dataset
from vqa_model import VQAModel
def train():
    model = VQAModel(18248, 300)
    optim = torch.optim.Adam(model.grad_params())
    dataset = Dataset('', 'train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    for f, bb, spat, obs, a, q, q_len, qid in loader:
        optim.zero_grad()
        pred=model(q,q_len, f,obs)
        loss = compute_multi_loss(pred, a)
       
        loss_item =  round(loss.item(),3)
        loss.backward()
        nn.utils.clip_grad_norm_(model.grad_params(), 5)
        optim.step()

        acc = compute_multi_acc(pred, a)
        print(acc)



def compute_multi_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).to(labels.device)
    loss *= labels.size(1)
    return loss

def compute_multi_acc(logits, labels):
    logit_max = logits.max(dim = 1)[1] # argmax
    logit_max.unsqueeze_(1).to(labels.device)
    one_hots = torch.zeros(*labels.size()).to(labels.device)
    one_hots.scatter_(1, logit_max, 1)
    acc = (one_hots * labels).sum() / labels.size(0) * 100
    return acc.item()

train()